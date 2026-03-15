import sys
import os
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
    # Filter for lines that actually look like map rows
    valid_lines = [line.strip() for line in lines if set(line.strip()).issubset({'.', '#', 'A', '0', '-', '*', 'M'})]
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
def perturb_environment(dsl_code, current_env, crash_report, visited_cells=None):
    print(f"\n[AI] Perturbing Environment to fix: {crash_report}")
    
    # Added a 60-second timeout to prevent the script from hanging on a slow AI response
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.5, timeout=60)
    
    current_map_str = map_to_string(current_env.state)
    
    # PREVENT PROMPT BLOAT: Only send a unique summary of the path
    path_info = ""
    if visited_cells:
        unique_cells = sorted(list(set(visited_cells)))
        if len(unique_cells) > 100:
            path_info = f"The robot visited {len(unique_cells)} cells. Key coordinates: {unique_cells[:50]} ... {unique_cells[-50:]}"
        else:
            path_info = f"The robot visited these coordinates: {unique_cells}"

    prompt = f"""
CURRENT MAP:
{current_map_str}

ROBOT PROGRAM:
{dsl_code}

ROBOT PATH RECORDED (Y, X coordinates):
{path_info}

TASK: 
Generate a 14x22 ASCII map that is 100% solvable by the provided ROBOT PROGRAM. 
Because the program is rigid, you must explicitly "hardcode" the environment to match the robot's exact footprints.

STRICT RULES:
1. GRID SIZE: The output MUST be exactly 14 rows by 22 columns.
2. THE TUNNEL: The ROBOT PATH RECORDED shows exactly where the robot tries to walk. EVERY coordinate in this path MUST be an empty floor ('.') or a dust marker ('*'). You are forbidden from placing walls on the recorded path.
3. THE DUST: You MUST place exactly 10 dust markers ('*') directly ON the robot's recorded path so the robot is guaranteed to step on them.
4. THE OBSTACLES: Fill the empty space outside of the robot's path with walls ('#') to make it look like a maze, but do not block the path itself.
5. OUTER BORDER: The edges of the map must remain walls ('-').

OUTPUT FORMAT:
Output ONLY the 14 lines of ASCII characters. Do not include any conversational text, explanations, or markdown formatting blocks. Just the raw text grid.
"""
    
    try:
        print("[Debug] Calling Ollama...")
        response = llm.invoke(prompt)
        print("[Debug] Received response from Ollama.")
        return parse_llm_map(response)
    except Exception as e:
        print(f"Ollama Error (likely timeout or connection): {e}")
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

    # 2. PARSE DSL
    dsl_parser = KarelDSL()
    try:
        program_node = dsl_parser.parse_str_to_node(dsl_code)
    except Exception as e:
        print(f"DSL Parse Error: {e}")
        return

    # 3. INITIALIZE ENVIRONMENT
    print("[Start] Initializing Base Environment...")
    env_args = {'env_height': 14, 'env_width': 22}
    task = CleanHouse(seed=0, env_args=env_args)
    env = task.initial_environment
    
    # 4. ITERATIVE REPAIR LOOP
    current_layout = None 
    
    for attempt in range(1, 6):
        print(f"\n--- Iteration {attempt}/5: Tracing & Testing ---")
        
        if current_layout:
             env_args['layout'] = current_layout
             # We use a unique seed for marker placement, but the layout is now AI-generated
             task = CleanHouse(seed=42+attempt, env_args=env_args)
             env = task.initial_environment

        sim_env = copy.deepcopy(env)
        step_gen = program_node.run_generator(sim_env)
        
        steps = 0
        crash_reason = None
        terminated = False
        reward = 0.0
        visited_cells = []
        
        try:
            # Record start pos
            if hasattr(sim_env, 'hero_pos'):
                visited_cells.append(tuple(sim_env.hero_pos))

            MAX_STEPS = 500 # Safety guard against infinite DSL loops
            for _ in step_gen:
                steps += 1
                if steps > MAX_STEPS:
                    crash_reason = "Timeout: Robot trapped in infinite loop."
                    break
                
                if hasattr(sim_env, 'hero_pos'):
                    visited_cells.append(tuple(sim_env.hero_pos))
                
                terminated, reward = task.get_reward(sim_env)
                if terminated:
                    if reward == task.crash_penalty:
                        crash_reason = f"Robot crashed into a wall at step {steps}."
                    break
            
            if not terminated: 
                terminated, reward = task.get_reward(sim_env)
            
            if reward >= 1.0:
                print("\n[SUCCESS] Environment solvable!")
                print(map_to_string(sim_env.state))
                break
            elif crash_reason is None:
                crash_reason = f"Incomplete: Reward was only {reward:.2f} (missed markers)."
                
        except Exception as e:
            crash_reason = f"Runtime Crash: {e}"

        print(f"[FAIL] {crash_reason}")
        print(f"   -> Robot visited {len(set(visited_cells))} unique cells.")
        
        # --- PERTURB ---
        current_layout = perturb_environment(dsl_code, env, crash_reason, visited_cells)
        
        if not current_layout:
            print("Failed to acquire map from LLM. Aborting.")
            break

if __name__ == "__main__":
    main()