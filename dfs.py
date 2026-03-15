import random
from langchain_ollama import OllamaLLM

# Setup the Model (We recommend qwen2.5-coder:7b for this specific task)
MODEL_NAME = "qwen2.5-coder:7b" 
NUM_SEEDS = 32

# The Hardcoded Map (0 = Empty, 1 = Wall)
world_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Direction vectors: 0=North, 1=East, 2=South, 3=West
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def get_turn_commands(current_facing, target_facing):
    """Calculates the Karel turns needed to face a new direction."""
    diff = (target_facing - current_facing) % 4
    if diff == 0: return []
    elif diff == 1: return ["turnRight"]
    elif diff == 2: return ["turnLeft", "turnLeft"]
    elif diff == 3: return ["turnLeft"]

def generate_randomized_dfs_path():
    """Runs a DFS that physically translates into Karel robot movements."""
    visited = set()
    commands = []
    
    # Start exactly where the evaluator spawns the robot
    start_y, start_x, start_facing = 1, 13, 2 # 2 = South
    
    def dfs(y, x, facing):
        visited.add((y, x))
        # Always check for dust on the current tile
        commands.append("IF c( markersPresent c) i( pickMarker i)")
        
        # SHUFFLE neighbors to guarantee a unique algorithm every time
        neighbors = [0, 1, 2, 3]
        random.shuffle(neighbors)
        
        for target_facing in neighbors:
            dy, dx = DIRS[target_facing]
            ny, nx = y + dy, x + dx
            
            # If the neighbor is an empty floor and hasn't been visited
            if 0 <= ny < 14 and 0 <= nx < 22 and world_map[ny][nx] == 0 and (ny, nx) not in visited:
                # 1. Turn and Move to neighbor
                commands.extend(get_turn_commands(facing, target_facing))
                commands.append("move")
                facing = target_facing
                
                # 2. Recurse (Robot explores that branch)
                facing = dfs(ny, nx, facing)
                
                # 3. BACKTRACK: The robot must physically return to (y, x)
                back_facing = (target_facing + 2) % 4 # Opposite direction
                commands.extend(get_turn_commands(facing, back_facing))
                commands.append("move")
                facing = back_facing
                
        return facing

    dfs(start_y, start_x, start_facing)
    return " ".join(commands)

def compress_with_llm(flat_sequence, index):
    """Uses Ollama to compress the flat sequence into a WHILE/REPEAT DSL program."""
    print(f"\n[AI] Compressing Seed {index}/{NUM_SEEDS}...")
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.5)
    
    prompt = f"""
    You are an expert compiler for Karel DSL.
    
    I have a brute-force, flat sequence of robot commands that perfectly solves a maze.
    However, it is too long. I need you to COMPRESS it into a generalized algorithm.
    
    THE FLAT SEQUENCE:
    {flat_sequence[:2000]} ... [TRUNCATED]
    
    YOUR TASK:
    Convert the underlying logic of searching a room into a structured Karel DSL program.
    Find patterns and use WHILE loops and IF statements to make a compact, smart algorithm.
    
    VALID COMMANDS: move, turnLeft, turnRight, pickMarker
    VALID CONDITIONS: markersPresent, noMarkersPresent, frontIsClear, leftIsClear, rightIsClear
    
    SYNTAX EXAMPLES:
    - WHILE c( frontIsClear c) w( move w)
    - IF c( markersPresent c) i( pickMarker i)
    - IFELSE c( leftIsClear c) i( turnLeft move i) ELSE e( turnRight move e)
    
    CRITICAL: Output ONLY the final `DEF run m( ... )` string. No explanations.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None

def main():
    print("=== Phase 1 & 2 & 3: DFS Path Generation to LLM Compression ===")
    
    generated_seeds = []
    
    for i in range(1, NUM_SEEDS + 1):
        # Step 1 & 2: Generate unique randomized DFS path
        flat_path = generate_randomized_dfs_path()
        print(f"\nSeed {i}: Generated flat DFS path with {len(flat_path.split())} instructions.")
        
        # Step 3: Send to LLM for pattern compression
        compressed_dsl = compress_with_llm(flat_path, i)
        
        if compressed_dsl:
            print(f"Result: {compressed_dsl[:100]}...")
            generated_seeds.append(compressed_dsl)
            
    # Save the 32 seeds to a file so they can be injected into the Hill Climbing evaluator
    with open("dfs_seeds.json", "w") as f:
        import json
        json.dump(generated_seeds, f, indent=4)
        
    print(f"\n[SUCCESS] Saved {len(generated_seeds)} compressed DFS algorithms to dfs_seeds.json!")
    print("Ready for Step 4: Inject these into the Hill Climbing Search.")

if __name__ == "__main__":
    main()