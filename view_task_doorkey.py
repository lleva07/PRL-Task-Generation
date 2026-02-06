

# import sys
# import os
# import re
# import numpy as np
# from collections import deque
# from langchain_community.llms import Ollama
# import prog_policies.utils 
# from prog_policies.karel_tasks.door_key import DoorKey
# from prog_policies.karel.dsl import KarelDSL

# MODEL_NAME = "llama3.2" 

# def parse_ascii_map(ascii_str, target_h=14, target_w=22):
#     """
#     Robust Parser: Converts a text block into a Karel Map.
#     """
#     lines = ascii_str.strip().split('\n')
#     grid = []
    
#     # 1. Filter for valid map lines (ignore text like "Here is the map:")
#     valid_lines = [line.strip() for line in lines if set(line.strip()).issubset({'.', '#', 'A', '0', '-'})]
    
#     # If filter failed, just take the raw lines that look long enough
#     if len(valid_lines) < 5:
#         valid_lines = lines

#     # 2. Convert to List and Fix Dimensions
#     for r in range(target_h):
#         new_row = []
#         if r < len(valid_lines):
#             # Take the line from the LLM
#             line_chars = list(valid_lines[r])
#         else:
#             # LLM ran out of lines? Add an empty row.
#             line_chars = []
            
#         for c in range(target_w):
#             if c < len(line_chars):
#                 char = line_chars[c]
#                 # Normalize symbols
#                 if char in ['0', '#', 'X']: 
#                     new_row.append('-') # Wall for Engine
#                 else: 
#                     new_row.append(0)   # Empty for Engine
#             else:
#                 # Pad missing columns with Empty space
#                 new_row.append(0)
                
#         # FORCE WALLS on borders (Crucial for Solvability)
#         new_row[0] = '-'
#         new_row[-1] = '-'
#         grid.append(new_row)
        
#     # Force Top/Bottom borders
#     grid[0] = ['-'] * target_w
#     grid[-1] = ['-'] * target_w
        
#     return grid

# def get_llm_map():
#     print(f"\n[1/3] Contacting Ollama ({MODEL_NAME})...")
#     llm = Ollama(model=MODEL_NAME, temperature=0.6) # Increased temp for variety
    
#     # NEW PROMPT: Focus on obstacles, not structure
#     prompt = """
#     Generate a 14x22 ASCII Grid Map.
#     '.' = Empty Space
#     '#' = Wall or Obstacle

#     Instructions:
#     1. Draw a large empty room (14 rows, 22 cols).
#     2. Place random '#' obstacles scattered inside.
#     3. Do NOT draw a vertical dividing line.
#     4. Do NOT draw keys or doors. 
#     5. Just draw a room with some scattered debris/walls inside.

#     Example Output format:
#     ######################
#     #.........#..........#
#     #...#...........#....#
#     #.......#.....#......#
#     ######################
#     """
    
#     try:
#         response = llm.invoke(prompt)
#         print(f"--- Raw AI Output ({len(response)} chars) ---")
#         return parse_ascii_map(response)
#     except Exception as e:
#         print(f"Ollama Error: {e}")
#         return None

# def check_solvability(env, task=None):
#     state = env.state
#     height, width = state.shape[1], state.shape[2]
#     walls = state[4, :, :].copy() # Copy so we can modify it virtually
    
#     start_r, start_c = env.hero_pos[0], env.hero_pos[1]
    
#     # --- PHASE 1: FIND THE KEY ---
#     # We use standard walls. Door is closed.
#     queue = deque([(start_r, start_c)])
#     visited = set([(start_r, start_c)])
#     key_found = False
    
#     while queue:
#         r, c = queue.popleft()
        
#         # Check if we stepped on the key
#         if task and (r, c) == task.key_cell:
#             key_found = True
#             break
            
#         for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < height and 0 <= nc < width:
#                 if not walls[nr, nc] and (nr, nc) not in visited:
#                     visited.add((nr, nc))
#                     queue.append((nr, nc))
    
#     if not key_found:
#         print("\n[FAIL] IMPOSSIBLE. Robot cannot reach the Key.")
#         return False
        
#     print("   -> Path to Key: CONFIRMED.")
    
#     # --- PHASE 2: FIND THE GOAL ---
#     # Now we assume the Agent is AT the Key, and the Door is OPEN.
    
#     # Virtual Unlock: Remove door walls
#     if task:
#         for (dr, dc) in task.door_cells:
#             walls[dr, dc] = False
            
#     # Reset Search from Key location
#     queue = deque([task.key_cell])
#     visited = set([task.key_cell])
#     goal_found = False
    
#     while queue:
#         r, c = queue.popleft()
        
#         if task and (r, c) == task.end_marker_cell:
#             goal_found = True
#             break
            
#         for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < height and 0 <= nc < width:
#                 if not walls[nr, nc] and (nr, nc) not in visited:
#                     visited.add((nr, nc))
#                     queue.append((nr, nc))

#     if not goal_found:
#         print("\n[FAIL] IMPOSSIBLE. Robot can get Key, but cannot reach Goal.")
#         return False

#     print("\n[PASS] SOLVABLE! Agent -> Key -> (Door Opens) -> Goal.")
#     return True

# def print_karel_grid(env, task=None): # Added task argument
#     state = env.state
#     _, height, width = state.shape
#     print(f"\n--- MAP VISUALIZATION ({height}x{width}) ---")
#     print("Legend: A=Agent, K=Key, G=Goal, #=Wall, D=Locked Door, .=Empty") # Added Legend

#     for y in range(height):
#         row_str = ""
#         for x in range(width):
#             # 1. Agent
#             if np.any(state[0:4, y, x]): 
#                 char = "A" 
            
#             # 2. Walls & Door (Need 'task' object to find door)
#             elif state[4, y, x]: 
#                 if task and hasattr(task, 'door_cells') and (y, x) in task.door_cells:
#                     char = "D"
#                 else:
#                     char = "#"
            
#             # 3. Markers (Need 'task' object to distinguish Key vs Goal)
#             elif np.any(state[6:, y, x]): 
#                 if task and hasattr(task, 'key_cell') and (y, x) == task.key_cell:
#                     char = "K"
#                 elif task and hasattr(task, 'end_marker_cell') and (y, x) == task.end_marker_cell:
#                     char = "G"
#                 else:
#                     char = "*"  # Unknown marker
            
#             # 4. Empty
#             else: 
#                 char = "."
            
#             row_str += char + " " 
#         print(row_str)
#     print("-" * (width * 2))

# def main():
#     final_map = get_llm_map()
#     if not final_map: return

#     print("\n[2/3] Loading into Environment...")
#     env_args = {'env_height': 14, 'env_width': 22, 'layout': final_map}
    
#     try:
#         task = DoorKey(seed=0, env_args=env_args)
#         env = task.initial_environment
#         print_karel_grid(env, task=task)
        
#         print("\n[3/3] Checking Solvability...")
#         check_solvability(env, task = task)
#     except Exception as e:
#         print(f"Validation Failed: {e}")


# def create_doorkey_task_from_llm(seed: int = 0, use_llm: bool = True):
#     """
#     Create a DoorKey task using an LLM-generated layout (if available).
#     Returns the instantiated task and its initial environment.
#     """
#     layout = None
#     if use_llm:
#         layout = get_llm_map()

#     # If LLM failed, create a minimal border-only layout
#     if not layout:
#         print("Using fallback empty layout for DoorKey (borders only)")
#         target_h, target_w = 14, 22
#         layout = [[0 for _ in range(target_w)] for _ in range(target_h)]
#         for r in range(target_h):
#             layout[r][0] = '-'
#             layout[r][-1] = '-'
#         layout[0] = ['-'] * target_w
#         layout[-1] = ['-'] * target_w

#     env_args = {'env_height': 14, 'env_width': 22, 'layout': layout}
#     try:
#         # Do NOT modify the LLM layout here. DoorKey will add the dividing wall
#         # and door cells itself so the two modules stay in sync.
#         task = DoorKey(seed=seed, env_args=env_args)
#         env = task.initial_environment
#         return task, env
#     except Exception as e:
#         print(f"Failed to create DoorKey task: {e}")
#         return None, None


# def run_program_on_task(program_str: str, task, verbose: bool = True):
#     """
#     Parse a DSL program string and evaluate it on the given task.
#     Returns the reward (float). If verbose, prints basic diagnostics.
#     """
#     if task is None:
#         raise ValueError("Task is None")

#     dsl = KarelDSL()
#     try:
#         program_node = dsl.parse_str_to_node(program_str)
#     except Exception as e:
#         print(f"Failed to parse program string into DSL node: {e}")
#         raise

#     # Evaluate (this will run the program on a fresh copy of the environment)
#     reward = task.evaluate_program(program_node)
#     if verbose:
#         print(f"Program evaluated. Reward: {reward}")
#     return reward


# if __name__ == "__main__":
#     # CLI: python view_task_doorkey.py /path/to/program.dsl
#     if len(sys.argv) == 2:
#         program_path = sys.argv[1]
#         if not os.path.exists(program_path):
#             print(f"Program file not found: {program_path}")
#             sys.exit(2)
#         program_text = open(program_path).read().strip()
#         task, env = create_doorkey_task_from_llm(seed=0, use_llm=True)
#         if task is None:
#             print("Could not create DoorKey task. Exiting.")
#             sys.exit(3)
#         print_karel_grid(env, task = task)
#         print("Running program on generated DoorKey environment...")
#         run_program_on_task(program_text, task, verbose=True)
#     else:
#         # Fallback to original main if no program provided
#         main()
import copy
import sys
import os
import random # <--- Added for variety
import time
import numpy as np
from langchain_community.llms import Ollama
import prog_policies.utils
from prog_policies.karel_tasks.door_key import DoorKey
from prog_policies.karel.dsl import KarelDSL

MODEL_NAME = "llama3.2" 

def parse_ascii_map(ascii_str, target_h=14, target_w=22):
    """Parses LLM output into a grid."""
    lines = ascii_str.strip().split('\n')
    # Filter only lines that look like map rows
    valid_lines = [line.strip() for line in lines if set(line.strip()).issubset({'.', '#', 'A', '0', '-'})]
    if len(valid_lines) < 5: valid_lines = lines # Fallback

    grid = []
    for r in range(target_h):
        new_row = []
        line_chars = list(valid_lines[r]) if r < len(valid_lines) else []
        for c in range(target_w):
            char = line_chars[c] if c < len(line_chars) else '.'
            # Treat # and - as walls, everything else as empty
            if char in ['#', '-', 'X']: new_row.append('-') 
            else: new_row.append(0)
        grid.append(new_row)
    return grid

def clean_dsl_program(program_str):
    """
    Nuclear Option: Cleans the string using simple replacement, no Regex.
    """
    # 1. Brute force remove source tags 0 through 99
    # This is "dumb" but guaranteed to work without syntax errors.
    for i in range(100):
        tag = f""
        program_str = program_str.replace(tag, "")
    
    # 2. Flatten and fix spacing
    program_str = program_str.replace('\n', ' ').replace('\t', ' ')
    tokens = ['m(', 'm)', 'c(', 'c)', 'w(', 'w)', 'i(', 'i)', 'e(', 'e)', 'r(', 'r)']
    for token in tokens:
        program_str = program_str.replace(token, f" {token} ")
        
    return ' '.join(program_str.split())

def get_llm_map():
    print(f"\n[1/3] Contacting Ollama ({MODEL_NAME})...")
    llm = Ollama(model=MODEL_NAME, temperature=0.8) # High temp for chaos
    
    prompt = """
    Generate a 14x22 ASCII Map.
    Characters: '.' (Empty), '#' (Obstacle).
    
    Instructions:
    1. Draw a room full of random obstacles, pillars, and debris.
    2. Make it messy.
    3. Do NOT draw a vertical dividing line (the code does that).
    4. Do NOT draw keys/doors (the code does that).
    
    Example:
    ######################
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

def print_karel_grid(env, task=None, step_num=None):
    state = env.state
    _, height, width = state.shape
    
    header = f"--- STATE VISUALIZATION"
    if step_num is not None: header += f" (Step {step_num})"
    print(f"\n{header} ---")
    
    # Check for crash (optional diagnostic)
    if hasattr(env, 'succeeded') and env.succeeded: print("STATUS: SUCCESS!")
    
    for y in range(height):
        row_str = ""
        for x in range(width):
            # Agent
            if np.any(state[0:4, y, x]): 
                dirs = ['^', '>', 'v', '<'] # N, E, S, W
                facing_idx = np.argmax(state[0:4, y, x])
                char = dirs[facing_idx] # Show facing direction
            
            # Walls & Door
            elif state[4, y, x]: 
                if task and (y, x) in task.door_cells: char = "D"
                else: char = "#"
            
            # Markers
            elif np.any(state[6:, y, x]): 
                if task and (y, x) == task.key_cell: char = "K"
                elif task and (y, x) == task.end_marker_cell: char = "G"
                else: char = "*" 
            else: 
                char = "."
            row_str += char + " " 
        print(row_str)

def run_debug_execution(program_str, task):
    """Runs the program step-by-step on the visualized map."""
    print("\n[DEBUG] Starting Step-by-Step Execution...")
    
    dsl = KarelDSL()
    
    print(f"[DEBUG] Raw Program: {program_str[:50]}...")
    program_str = clean_dsl_program(program_str)
    print(f"[DEBUG] Cleaned Program: {program_str[:50]}...")
    
    try:
        program_node = dsl.parse_str_to_node(program_str)
    except Exception as e:
        print(f"Parser Error: {e}")
        return

    # 1. Reset Task Logic (Flags like door_locked)
    task.reset_environment()
    
    # 2. CRITICAL: Use the EXACT map we just visualized
    # We copy it so we don't accidentally ruin the original if we run this twice.
    env = copy.deepcopy(task.initial_environment)
    
    try:
        # 3. Create a Generator to step through the code
        # This allows us to update the "Door Logic" after every move
        step_gen = program_node.run_generator(env)
        
        steps = 0
        terminated = False
        reward = 0.0
        
        for _ in step_gen:
            steps += 1
            # Update the Game Physics (Check if Key picked -> Open Door)
            terminated, reward = task.get_reward(env)
            
            if terminated:
                break
        
        # Final Reward Check
        if not terminated:
             _, reward = task.get_reward(env)
             
        print(f"\n[RESULT] Final Reward: {reward}")
        
        if reward == 1.0:
            print("Outcome: SUCCESS (Goal Reached)")
        elif reward == task.crash_penalty:
            print("Outcome: CRASH (Hit a wall or illegal move)")
        else:
            print("Outcome: INCOMPLETE (Timed out or didn't reach goal)")
            
        print("Final Robot Position:")
        print_karel_grid(env, task=task, step_num="FINAL")
        
    except Exception as e:
        print(f"Runtime Error: {e}")
        # Print where it died
        print("Crash State:")
        print_karel_grid(env, task=task, step_num="CRASH")

def main():
    # 1. Use a RANDOM seed to get different map layouts/locations
    seed = random.randint(0, 100000)
    print(f"Using Random Seed: {seed}")
    
    llm_map = get_llm_map()
    env_args = {'env_height': 14, 'env_width': 22, 'layout': llm_map}
    
    try:
        task = DoorKey(seed=seed, env_args=env_args)
        print("\n[Initial State]")
        print_karel_grid(task.initial_environment, task=task)
        
        # CLI Argument handling for DSL file
        if len(sys.argv) == 2:
            prog_path = sys.argv[1]
            if os.path.exists(prog_path):
                code = open(prog_path).read().strip()
                run_debug_execution(code, task)
            else:
                print("File not found.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()