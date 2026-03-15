import sys
import os
import re
import copy
import numpy as np
from collections import deque
from langchain_community.llms import Ollama
import prog_policies.utils 
from prog_policies.karel_tasks.clean_house import CleanHouse
from prog_policies.karel.dsl import KarelDSL

MODEL_NAME = "llama3.2" 

def ensure_connectivity(grid):
    """
    SAFETY DRILL: Ensures every empty spot is reachable from the center.
    If a spot is isolated, it drills through walls until connected.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # 1. Find a valid start point (first empty spot)
    start = None
    all_empty_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                if start is None: start = (r, c)
                all_empty_cells.append((r, c))
    
    if not start: return grid # No empty space at all?

    # 2. Flood Fill to find reachable cells
    queue = deque([start])
    reachable = set([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0 and (nr, nc) not in reachable:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))

    # 3. Check for isolated cells
    unreachable = [cell for cell in all_empty_cells if cell not in reachable]
    
    # 4. If found, DRILL HOLES
    if unreachable:
        print(f"   [Auto-Fix] Drilling holes to connect {len(unreachable)} isolated cells...")
        for r, c in unreachable:
            # Drill neighbors until we hit a reachable spot or another empty spot
            # This is a naive fix: just delete walls around isolated cells
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 < nr < rows-1 and 0 < nc < cols-1: # Don't drill border
                    if grid[nr][nc] == '-': # If it's a wall
                        grid[nr][nc] = 0    # SMASH IT
                        
    return grid

def parse_ascii_map(ascii_str, target_h=14, target_w=22):
    """Robust Parser: Converts a text block into a Karel Map."""
    lines = ascii_str.strip().split('\n')
    valid_lines = [line.strip() for line in lines if set(line.strip()).issubset({'.', '#', 'A', '0', '-'})]
    if len(valid_lines) < 5: valid_lines = lines

    grid = []
    for r in range(target_h):
        new_row = []
        line_chars = list(valid_lines[r]) if r < len(valid_lines) else []
        for c in range(target_w):
            char = line_chars[c] if c < len(line_chars) else '.'
            if char in ['0', '#', 'X', '-']: new_row.append('-') 
            else: new_row.append(0)
        new_row[0] = '-'; new_row[-1] = '-' # Force borders
        grid.append(new_row)
    grid[0] = ['-'] * target_w; grid[-1] = ['-'] * target_w
    
    # Run the Safety Drill
    grid = ensure_connectivity(grid)
    
    return grid

def get_llm_map(dsl_code=None):
    print(f"\n[1/3] Contacting Ollama ({MODEL_NAME})...")
    llm = Ollama(model=MODEL_NAME, temperature=0.6)
    
    # --- UPDATED PROMPT: "Open Warehouse" Strategy ---
    base_instructions = """
    Generate a 14x22 ASCII Grid Map.
    '.' = Empty Space
    '#' = Wall/Obstacle

    CRITICAL RULES:
    1. Draw a LARGE OPEN WAREHOUSE with scattered pillars.
    2. DO NOT DRAW ROOMS. DO NOT DRAW LONG WALLS.
    3. Ensure 100% connectivity. No empty spot should be walled off.
    4. Every wall '#' must have open space around it.
    5. Output ONLY the map.
    """

    if dsl_code:
        print("   -> Using DSL Code to guide Map Generation...")
        prompt = f"""
        {base_instructions}
        
        I have a robot program that looks like this:
        {dsl_code}
        
        Design the obstacle placement specifically for this code.
        - Ensure the robot can move freely.
        - Do not block the robot's path with solid lines.
        - Keep it open and spacious.
        """
    else:
        prompt = f"""
        {base_instructions}
        Example:
        ######################
        #....................#
        #...#.......#........#
        #.......#.......#....#
        #...#.......#........#
        ######################
        """
    
    try:
        response = llm.invoke(prompt)
        return parse_ascii_map(response)
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None

def print_karel_grid(env, step_num=None):
    state = env.state
    _, height, width = state.shape
    header = f"--- MAP VISUALIZATION ({height}x{width})"
    if step_num is not None: header += f" (Step {step_num})"
    print(f"\n{header} ---")
    for y in range(height):
        row_str = ""
        for x in range(width):
            if state[4, y, x]: char = "#"      
            elif np.any(state[0:4, y, x]): 
                dirs = ['^', '>', 'v', '<']
                char = dirs[np.argmax(state[0:4, y, x])]
            elif np.any(state[6:, y, x]): char = "*"  
            else: char = "."
            row_str += char + " " 
        print(row_str)
    print("-" * (width * 2))

def run_dsl_program(program_str, task):
    """Parses and runs the DSL program."""
    print("\n[DEBUG] Parsing Program...")
    dsl = KarelDSL()
    try:
        program_node = dsl.parse_str_to_node(program_str)
    except Exception as e:
        print(f"Parser Error: {e}")
        return

    task.reset_environment()
    env = copy.deepcopy(task.initial_environment)
    print("\n[DEBUG] Running Program Step-by-Step...")
    
    try:
        step_gen = program_node.run_generator(env)
        steps = 0
        terminated = False
        reward = 0.0
        for _ in step_gen:
            steps += 1
            terminated, reward = task.get_reward(env)
            if terminated: break
        
        if not terminated: _, reward = task.get_reward(env)
             
        print(f"\n[RESULT] Final Reward: {reward}")
        if reward == 1.0: print("Outcome: SUCCESS")
        elif reward == task.crash_penalty: print("Outcome: CRASH")
        else: print(f"Outcome: INCOMPLETE (Reward: {reward})")
            
        print("Final State:")
        print_karel_grid(env, step_num="FINAL")
        
    except Exception as e:
        print(f"Runtime Error: {e}")

def main():
    # 1. READ DSL FILE FIRST
    dsl_code = None
    dsl_file = None
    
    if len(sys.argv) == 2:
        dsl_file = sys.argv[1]
        if os.path.exists(dsl_file):
            print(f"[Input] Reading DSL solution from: {dsl_file}")
            with open(dsl_file, 'r') as f:
                dsl_code = f.read().strip()
        else:
            print(f"File not found: {dsl_file}")
            return

    # # 2. GENERATE MAP (Conditioned on DSL if available)
    # final_map = get_llm_map(dsl_code)
    # if not final_map: return

    # print("\n[2/3] Loading into Environment...")
    # env_args = {'env_height': 14, 'env_width': 22, 'layout': final_map}
    
    print("\n[2/3] Loading into Environment...")
    # Remove 'layout' so it uses the hardcoded CleanHouse map
    env_args = {'env_height': 14, 'env_width': 22}
    
    try:
        task = CleanHouse(seed=0, env_args=env_args)
        
        # Show Initial State
        print_karel_grid(task.initial_environment, step_num="START")
        
        # 3. RUN PROGRAM (If we have one)
        if dsl_code:
            print(f"\n[3/3] Testing generated environment with: {dsl_file}")
            run_dsl_program(dsl_code, task)
        else:
            print("\nTo generate a solution-specific map, use: python3 view_task.py <your_file.dsl>")

    except Exception as e:
        print(f"Validation Failed: {e}")

if __name__ == "__main__":
    main()