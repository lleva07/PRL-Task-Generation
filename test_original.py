import sys
import copy
import numpy as np
import prog_policies.utils 
from prog_policies.karel_tasks.clean_house import CleanHouse
from prog_policies.karel.dsl import KarelDSL

def print_grid(env, steps):
    state = env.state
    _, height, width = state.shape
    print(f"\n--- Step {steps} ---")
    for y in range(height):
        row_str = ""
        for x in range(width):
            if state[4, y, x]: char = "#"      # Wall
            elif np.any(state[0:4, y, x]): 
                dirs = ['^', '>', 'v', '<']
                char = dirs[np.argmax(state[0:4, y, x])] # Robot Direction
            elif state[6, y, x]: char = "*"    # Dust Marker
            else: char = "."
            row_str += char + " " 
        print(row_str)

def main():
    # 1. LOAD DSL
    dsl_file = "cleanhouse.dsl"
    try:
        with open(dsl_file, 'r') as f:
            dsl_code = f.read().strip()
        print(f"[1/3] Loaded DSL: {dsl_file}")
    except FileNotFoundError:
        print(f"Error: {dsl_file} not found.")
        return

    # 2. INITIALIZE ORIGINAL SEED 0 ENVIRONMENT
    print("[2/3] Initializing CleanHouse with Seed 0...")
    env_args = {'env_height': 14, 'env_width': 22}
    task = CleanHouse(seed=0, env_args=env_args)
    env = task.initial_environment

    # 3. PARSE AND RUN
    print("[3/3] Executing Program...")
    dsl_parser = KarelDSL()
    program_node = dsl_parser.parse_str_to_node(dsl_code)
    
    steps = 0
    max_steps = 100
    print_grid(env, steps)

    try:
        step_gen = program_node.run_generator(env)
        for _ in step_gen:
            steps += 1
            print_grid(env, steps)
            
            terminated, reward = task.get_reward(env)
            
            if terminated:
                if reward == task.crash_penalty:
                    print(f"\n[CRASH] Robot hit a wall at step {steps}.")
                else:
                    print(f"\n[SUCCESS] Task finished! Reward: {reward}")
                break
            
            if steps >= max_steps:
                print("\n[TIMEOUT] Robot exceeded max steps.")
                break
        
        if steps == 0:
            print("\n[IDLE] Robot did not move. Checking condition...")
            _, reward = task.get_reward(env)
            print(f"Final Reward: {reward}")

    except Exception as e:
        print(f"\n[ERROR] Runtime failure: {e}")

if __name__ == "__main__":
    main()