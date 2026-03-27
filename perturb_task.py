import sys
import copy
import numpy as np
from langchain_ollama import OllamaLLM

# --- CRITICAL FIX: Import utils first to prevent circular error ---
import prog_policies.utils 
from prog_policies.karel_tasks.clean_house import CleanHouse
from prog_policies.karel.dsl import KarelDSL

MODEL_NAME = "qwen2.5-coder:7b" 

# --- 1. MAP PARSER ---
def parse_llm_map(ascii_str, target_h=14, target_w=22):
    lines = ascii_str.strip().split('\n')
    valid_lines = [line.strip() for line in lines if set(line.strip()).issubset({'.', '#', 'A', '0', '-', '*', 'M', 'v', '^', '<', '>'})]
    if len(valid_lines) < 5: valid_lines = lines 

    grid = []
    for r in range(target_h):
        new_row = []
        line_chars = list(valid_lines[r]) if r < len(valid_lines) else []
        for c in range(target_w):
            char = line_chars[c] if c < len(line_chars) else '.'
            if char in ['#', '0', 'X', '-']: 
                new_row.append('-') # Wall
            elif char in ['*', 'M']:
                new_row.append(1)   # Marker/Dust
            else:
                new_row.append(0)   # Empty Floor
        new_row[0] = '-'; new_row[-1] = '-'
        grid.append(new_row)
    grid[0] = ['-'] * target_w; grid[-1] = ['-'] * target_w
    return grid

def map_to_string(state):
    _, h, w = state.shape
    output = ""
    for y in range(h):
        row = ""
        for x in range(w):
            if state[4, y, x]: row += "#"
            elif np.any(state[0:4, y, x]): row += "A" 
            elif state[6, y, x]: row += "*" # Dust/Marker Layer
            else: row += "."
        output += row + "\n"
    return output

# --- 2. THE PERTURBATION LOOP ---
def perturb_environment(dsl_code, current_env, crash_report):
    print(f"\n[AI] Perturbing Environment to fix: {crash_report}")
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.5, timeout=60)
    current_map_str = map_to_string(current_env.state)
    
    prompt = f"""
DSL SYNTAX CHEAT SHEET:
- m( ... m) : Main program block definition.
- w( ... w) : While loop.
- i( ... i) : If condition block.
- r( ... r) : Repeat loop.
- c( frontIsClear c) : Condition checking if no wall is in front.
- c( noMarkersPresent c) : Condition checking if tile has no dust.
- move, turnLeft, turnRight, pickMarker : Standard robot actions.

CURRENT MAP:
{current_map_str}

ROBOT PROGRAM:
{dsl_code}

CRASH/FAILURE REPORT:
{crash_report}

TASK:
You are a highly restricted grid-mutation algorithm. The robot failed on the CURRENT MAP. Your job is to make a map that allows the robot to progress further.

ABSOLUTE RULES:
1. GRID DIMENSIONS: The output MUST be exactly 14 rows by 22 columns. The outer border MUST remain '-'.
2. ONE CHANGE ONLY: You are FORBIDDEN from redrawing the map. You may ONLY make ONE or TWO small changes compared to the CURRENT MAP (e.g., remove one blocking wall '#', or move one dust marker '*' to a tile the robot is trapped on).
3. DO NOT MOVE THE ROBOT: Leave the 'A' exactly where it is.

OUTPUT FORMAT:
Output ONLY the 14 lines of ASCII characters. NO markdown tags, NO explanations. Just the raw grid."""
    
    try:
        response = llm.invoke(prompt)
        return parse_llm_map(response)
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 perturb_task.py <solution.dsl>")
        return

    # 1. LOAD DSL
    dsl_path = sys.argv[1]
    with open(dsl_path, 'r') as f:
        dsl_code = f.read().strip()
    print(f"[Input] Loaded DSL from {dsl_path}")

    dsl_parser = KarelDSL()
    program_node = dsl_parser.parse_str_to_node(dsl_code)

    # --- GREEDY ALGORITHM TRACKERS ---
    best_reward = -1.0
    best_layout = None
    stagnant_attempts = 0
    current_layout = None 
    
    # Increased loop to 30 to give the greedy algorithm time to work
    for attempt in range(1, 31):
        print(f"\n================================================")
        print(f"--- Iteration {attempt}/30: Tracing & Testing ---")
        
        env_args = {'env_height': 14, 'env_width': 22}
        if current_layout:
             env_args['layout'] = current_layout
             
        task = CleanHouse(seed=42+attempt, env_args=env_args)
        env = task.initial_environment
        
        # 3. PRINT CURRENT MAP TO TERMINAL
        current_map_str = map_to_string(env.state)
        print(f"TESTING THIS MAP:\n{current_map_str}")

        sim_env = copy.deepcopy(env)
        step_gen = program_node.run_generator(sim_env)
        
        steps = 0
        crash_reason = None
        terminated = False
        reward = 0.0
        
        try:
            MAX_STEPS = 800 # Increased to match your test_original run
            for _ in step_gen:
                steps += 1
                if steps > MAX_STEPS:
                    crash_reason = "Timeout: Robot trapped in infinite loop."
                    break
                
                terminated, reward = task.get_reward(sim_env)
                if terminated:
                    if reward == task.crash_penalty:
                        crash_reason = f"Robot crashed into a wall at step {steps}."
                    break
            
            if not terminated: 
                terminated, reward = task.get_reward(sim_env)
            
            if reward >= 1.0:
                print("\n[SUCCESS] Environment solvable! Final Reward: 1.0")
                break
            elif crash_reason is None:
                crash_reason = f"Incomplete: Reward was only {reward:.2f}."
                
        except Exception as e:
            crash_reason = f"Runtime Crash: {e}"

        print(f"[FAIL] {crash_reason}")
        
        # --- GREEDY LOGIC EVALUATION ---
        if reward > best_reward:
            print(f"[IMPROVEMENT] Reward increased from {best_reward:.2f} to {reward:.2f}! Saving layout.")
            best_reward = reward
            # Save layout (if it's the first run, save the base map)
            best_layout = copy.deepcopy(current_layout) if current_layout else parse_llm_map(current_map_str)
            stagnant_attempts = 0
        else:
            stagnant_attempts += 1
            print(f"[STAGNANT] Reward ({reward:.2f}) did not improve. Stagnant counter: {stagnant_attempts}/5")
            
        # --- REVERT LOGIC ---
        if stagnant_attempts >= 5:
            print(f"\n[REVERT TRIGGERED] 5 failed attempts in a row. Reverting to best map (Reward: {best_reward:.2f}).")
            current_layout = copy.deepcopy(best_layout)
            stagnant_attempts = 0
            # Ask LLM to perturb the *best* map again, but maybe a different way this time
            task_for_perturb = CleanHouse(seed=0, env_args={'env_height': 14, 'env_width': 22, 'layout': current_layout})
            env_for_perturb = task_for_perturb.initial_environment
            current_layout = perturb_environment(dsl_code, env_for_perturb, "REVERTED. Try a DIFFERENT small change than before to improve the path.")
            continue
        
        # --- STANDARD PERTURBATION ---
        current_layout = perturb_environment(dsl_code, env, crash_reason)
        
        if not current_layout:
            print("Failed to acquire map from LLM. Aborting.")
            break

if __name__ == "__main__":
    main()