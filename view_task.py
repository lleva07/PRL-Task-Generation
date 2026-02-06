import sys
import os
import re
import copy
import numpy as np
from collections import deque
from langchain_community.llms import Ollama
import prog_policies.utils 
from prog_policies.karel_tasks.clean_house import CleanHouse
# --- ADDED: Import the DSL Parser ---
from prog_policies.karel.dsl import KarelDSL

MODEL_NAME = "llama3.2" 

def parse_ascii_map(ascii_str, target_h=14, target_w=22):
    """
    Robust Parser: Converts a text block into a Karel Map.
    """
    lines = ascii_str.strip().split('\n')
    grid = []
    
    # 1. Filter for valid map lines (ignore text like "Here is the map:")
    valid_lines = [line.strip() for line in lines if set(line.strip()).issubset({'.', '#', 'A', '0', '-'})]
    
    # If filter failed, just take the raw lines that look long enough
    if len(valid_lines) < 5:
        valid_lines = lines

    # 2. Convert to List and Fix Dimensions
    for r in range(target_h):
        new_row = []
        if r < len(valid_lines):
            # Take the line from the LLM
            line_chars = list(valid_lines[r])
        else:
            # LLM ran out of lines? Add an empty row.
            line_chars = []
            
        for c in range(target_w):
            if c < len(line_chars):
                char = line_chars[c]
                # Normalize symbols
                if char in ['0', '#', 'X']: 
                    new_row.append('-') # Wall for Engine
                else: 
                    new_row.append(0)   # Empty for Engine
            else:
                # Pad missing columns with Empty space
                new_row.append(0)
                
        # FORCE WALLS on borders (Crucial for Solvability)
        new_row[0] = '-'
        new_row[-1] = '-'
        grid.append(new_row)
        
    # Force Top/Bottom borders
    grid[0] = ['-'] * target_w
    grid[-1] = ['-'] * target_w
        
    return grid

def get_llm_map():
    print(f"\n[1/3] Contacting Ollama ({MODEL_NAME})...")
    
    # Temperature 0.4 = More focused, less random walls
    llm = Ollama(model=MODEL_NAME, temperature=0.4)
    
    # ASK FOR ASCII ART, NOT PYTHON LISTS
    prompt = """
    Generate a 14x22 Grid Map for a game.
    Use ASCII characters.
    '.' is Empty Space
    '#' is Wall
    
    Rules:
    1. Draw exactly 14 lines.
    2. Each line must have 22 characters.
    3. Make it look like a building floor plan with rooms.
    4. DO NOT WRITE CODE. Just the map.
    
    Example:
    ######################
    #....................#
    #...#######..........#
    #....................#
    ######################
    """
    
    try:
        response = llm.invoke(prompt)
        print(f"--- Raw AI Output ({len(response)} chars) ---")
        # Print a preview of what the AI sent
        print('\n'.join(response.split('\n')[:5]) + "...\n") 
        
        return parse_ascii_map(response)
        
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None

def check_solvability(env):
    state = env.state
    walls = state[4, :, :]
    # Get all markers
    markers = np.sum(state[6:, :, :], axis=0) > 0
    marker_coords = list(zip(*np.where(markers)))
    
    if not marker_coords: 
        print("[WARN] No markers found. (Technically Solvable)")
        return True

    # Get Robot
    start_r, start_c = env.hero_pos[0], env.hero_pos[1]
    
    # Safety: If LLM put a wall on the robot, remove it.
    if walls[start_r, start_c]:
        print("   -> Robot spawned in a wall! Clearing that spot...")
        walls[start_r, start_c] = False 
        # Update the real state so the game works
        env.state[4, start_r, start_c] = False

    # Flood Fill
    queue = deque([(start_r, start_c)])
    visited = set([(start_r, start_c)])
    height, width = walls.shape
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if not walls[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    unreachable = [m for m in marker_coords if m not in visited]
            
    if unreachable:
        print(f"\n[FAIL] IMPOSSIBLE. {len(unreachable)} markers are unreachable.")
        return False
    else:
        print(f"\n[PASS] SOLVABLE! Robot can reach all markers.")
        return True

def print_karel_grid(env, step_num=None): # --- ADDED step_num for tracing ---
    state = env.state
    _, height, width = state.shape
    
    # --- ADDED: Header handling ---
    header = f"--- MAP VISUALIZATION ({height}x{width})"
    if step_num is not None: header += f" (Step {step_num})"
    print(f"\n{header} ---")
    
    for y in range(height):
        row_str = ""
        for x in range(width):
            if state[4, y, x]: char = "#"      
            elif np.any(state[0:4, y, x]): char = "A" 
            elif np.any(state[6:, y, x]): char = "*"  
            else: char = "."
            row_str += char + " " 
        print(row_str)
    print("-" * (width * 2))

# --- ADDED: DSL Execution Function ---
def run_dsl_program(program_str, task):
    """
    Parses and runs the DSL program on the generated map.
    """
    print("\n[DEBUG] Parsing Program...")
    dsl = KarelDSL()
    
    try:
        # Parse exactly as written in the file (No Regex cleaning)
        program_node = dsl.parse_str_to_node(program_str)
    except Exception as e:
        print(f"Parser Error: {e}")
        return

    # 1. Reset and Copy Environment
    task.reset_environment()
    env = copy.deepcopy(task.initial_environment)
    
    print("\n[DEBUG] Running Program Step-by-Step...")
    
    try:
        step_gen = program_node.run_generator(env)
        steps = 0
        terminated = False
        reward = 0.0
        
        # Run until finished
        for _ in step_gen:
            steps += 1
            terminated, reward = task.get_reward(env)
            if terminated: break
        
        # Double check final reward state
        if not terminated: 
            _, reward = task.get_reward(env)
             
        print(f"\n[RESULT] Final Reward: {reward}")
        if reward == 1.0: print("Outcome: SUCCESS (Cleaned all markers!)")
        elif reward == task.crash_penalty: print("Outcome: CRASH (Hit a wall)")
        else: print(f"Outcome: INCOMPLETE (Reward: {reward})")
            
        print("Final State:")
        print_karel_grid(env, step_num="FINAL")
        
    except Exception as e:
        print(f"Runtime Error: {e}")
        print_karel_grid(env, step_num="CRASH")

def main():
    final_map = get_llm_map()
    if not final_map: return

    print("\n[2/3] Loading into Environment...")
    env_args = {'env_height': 14, 'env_width': 22, 'layout': final_map}
    
    try:
        task = CleanHouse(seed=0, env_args=env_args)
        env = task.initial_environment
        print_karel_grid(env)
        
        print("\n[3/3] Checking Solvability...")
        check_solvability(env)
        
        # --- ADDED: CLI Logic to Run Program ---
        if len(sys.argv) == 2:
            dsl_file = sys.argv[1]
            if os.path.exists(dsl_file):
                print(f"\n[4/4] Executing Program: {dsl_file}")
                with open(dsl_file, 'r') as f:
                    code = f.read().strip()
                run_dsl_program(code, task)
            else:
                print(f"File not found: {dsl_file}")
        else:
            print("\nTo run a program, use: python3 view_task.py <your_file.dsl>")

    except Exception as e:
        print(f"Validation Failed: {e}")

if __name__ == "__main__":
    main()