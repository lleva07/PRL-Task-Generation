# import sys
# import os
# import ast
# import re
# import json
# import numpy as np
# from collections import deque
# from langchain_community.llms import Ollama
# import prog_policies.utils 
# from prog_policies.karel_tasks.clean_house import CleanHouse

# # --- CONFIGURATION ---
# MODEL_NAME = "llama3.2" 

# def fix_map_dimensions(grid, target_h=14, target_w=22):
#     """
#     AUTO-REPAIR: Ensures the map is exactly 14x22.
#     """
#     # 1. Fix Height (Rows)
#     current_h = len(grid)
#     if current_h < target_h:
#         print(f"   -> Repairing Height: Adding {target_h - current_h} empty rows.")
#         for _ in range(target_h - current_h):
#             new_row = ['.'] * target_w # Use . for empty
#             new_row[0] = '#'; new_row[-1] = '#' # Use # for wall
#             grid.append(new_row)
#     elif current_h > target_h:
#         grid = grid[:target_h]
    
#     # 2. Fix Width (Columns)
#     for i in range(len(grid)):
#         # Force conversion to list in case of tuples
#         row = list(grid[i])
#         current_w = len(row)
#         if current_w < target_w:
#             row.extend(['.'] * (target_w - current_w))
#         elif current_w > target_w:
#             row = row[:target_w]
#         grid[i] = row
            
#     return grid

# def translate_map_for_karel(llm_grid):
#     """
#     TRANSLATOR: Converts LLM symbols to Karel Engine symbols.
#     LLM says: '.' (Empty), '#' (Wall) -> Engine expects: 0 (Empty), '-' (Wall)
#     """
#     karel_grid = []
#     for row in llm_grid:
#         new_row = []
#         for cell in row:
#             s_cell = str(cell)
#             # Logic: If it looks like a wall (0 or #), it's a wall.
#             if s_cell == '#' or s_cell == '0': 
#                 new_row.append('-') 
#             else:
#                 new_row.append(0)
#         karel_grid.append(new_row)
#     return karel_grid

# def get_llm_map():
#     print(f"\n[1/3] Contacting Ollama ({MODEL_NAME})...")
#     llm = Ollama(model=MODEL_NAME, temperature=0.6)
    
#     # Simplified Prompt: "Just give me the list"
#     prompt = """
#     Generate a 14x22 Grid Map.
#     Output ONLY a Python list of lists.
    
#     Legend:
#     '.' = Empty Space
#     '#' = Wall
    
#     Rules:
#     1. Borders must be '#'.
#     2. Output raw list only.
#     3. Do NOT write a script. Do NOT use "import".
    
#     Example:
#     [['#', '#', '#'], ['#', '.', '#']]
#     """
    
#     try:
#         response = llm.invoke(prompt)
#     except Exception as e:
#         print(f"Ollama Error: {e}")
#         return None

#     # --- BULLETPROOF PARSER ---
#     # 1. Regex to find the list block [[ ... ]] inside the text
#     # This ignores "Here is the code:" at the start.
#     match = re.search(r"\[\s*\[.*?\]\s*\]", response, re.DOTALL)
    
#     if match:
#         clean_str = match.group(0)
#         # 2. Dual Parse Strategy
#         try:
#             # Plan A: Try reading as Python List (Best for single quotes)
#             raw_map = ast.literal_eval(clean_str)
#             return fix_map_dimensions(raw_map)
#         except:
#             try:
#                 # Plan B: Try reading as JSON (Best for double quotes)
#                 raw_map = json.loads(clean_str)
#                 return fix_map_dimensions(raw_map)
#             except:
#                 pass

#     print("Could not extract map. The AI wrote a script instead of data.")
#     print("Raw Response snippet:", response[:200])
#     return None

# def check_solvability(env):
#     state = env.state
#     walls = state[4, :, :]
#     markers = np.sum(state[6:, :, :], axis=0) > 0
#     marker_coords = list(zip(*np.where(markers)))
    
#     if not marker_coords: 
#         print("Warning: No markers generated. Technically solvable.")
#         return True 

#     start_r, start_c = env.hero_pos[0], env.hero_pos[1]
    
#     if walls[start_r, start_c]:
#         print("\n[FAIL] Robot spawned INSIDE a wall.")
#         return False

#     queue = deque([(start_r, start_c)])
#     visited = set([(start_r, start_c)])
#     directions = [(-1,0), (1,0), (0,-1), (0,1)]
#     height, width = walls.shape
    
#     while queue:
#         r, c = queue.popleft()
#         for dr, dc in directions:
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < height and 0 <= nc < width:
#                 if not walls[nr, nc] and (nr, nc) not in visited:
#                     visited.add((nr, nc))
#                     queue.append((nr, nc))
    
#     unreachable = [m for m in marker_coords if m not in visited]
            
#     if unreachable:
#         print(f"\n[FAIL] Map is IMPOSSIBLE. Unreachable markers: {len(unreachable)}")
#         return False
#     else:
#         print(f"\n[PASS] Map is SOLVABLE.")
#         return True

# def print_karel_grid(env):
#     state = env.state
#     _, height, width = state.shape
#     print(f"\n--- MAP VISUALIZATION ({height}x{width}) ---")
#     for y in range(height):
#         row_str = ""
#         for x in range(width):
#             if state[4, y, x]: char = "#"      
#             elif np.any(state[0:4, y, x]): char = "A" 
#             elif np.any(state[6:, y, x]): char = "*"  
#             else: char = "."
#             row_str += char + " " 
#         print(row_str)
#     print("-" * (width * 2))

# def main():
#     llm_map = get_llm_map()
#     if not llm_map: return

#     print("   -> Translating symbols for Karel Engine...")
#     karel_ready_map = translate_map_for_karel(llm_map)

#     print("\n[2/3] Loading into Environment...")
#     env_args = {'env_height': 14, 'env_width': 22, 'layout': karel_ready_map}
    
#     try:
#         task = CleanHouse(seed=0, env_args=env_args)
#         env = task.initial_environment
#         print_karel_grid(env)
#         print("\n[3/3] Validating Solvability...")
#         check_solvability(env)
#     except Exception as e:
#         print(f"Validation Failed: {e}")

# if __name__ == "__main__":
#     main()

import sys
import os
import re
import numpy as np
from collections import deque
from langchain_community.llms import Ollama
import prog_policies.utils 
from prog_policies.karel_tasks.clean_house import CleanHouse

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

def print_karel_grid(env):
    state = env.state
    _, height, width = state.shape
    print(f"\n--- MAP VISUALIZATION ({height}x{width}) ---")
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
    except Exception as e:
        print(f"Validation Failed: {e}")

if __name__ == "__main__":
    main()