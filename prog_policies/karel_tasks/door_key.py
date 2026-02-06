from math import ceil
import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class DoorKey(BaseTask):
        
    def generate_initial_environment(self, env_args):
        # 1. SETUP
        run_args = env_args.copy()
        world_map = None
        if 'layout' in run_args and run_args['layout'] is not None:
            world_map = run_args.pop('layout')
        
        reference_env = KarelEnvironment(**run_args)
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        # 2. LAYER 1: THE LLM MAP
        if world_map:
            for y in range(env_height):
                for x in range(env_width):
                    if world_map[y][x] == '-':
                        state[4, y, x] = True
        else:
            state[4, :, 0] = True; state[4, :, env_width - 1] = True
            state[4, 0, :] = True; state[4, env_height - 1, :] = True

        # 3. LAYER 2: THE ENFORCER
        state[4, :, 0] = True; state[4, :, env_width - 1] = True
        state[4, 0, :] = True; state[4, env_height - 1, :] = True
        wall_column = ceil(env_width / 2)
        state[4, :, wall_column] = True
        
        # --- NEW 3x3 BULLDOZER FUNCTION ---
        def clear_safety_bubble(y, x):
            # Clears a full 3x3 square around the target
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y+dy, x+dx
                    if 0 < ny < env_height-1 and 0 < nx < env_width-1:
                        # Never delete the Middle Wall!
                        if nx != wall_column: 
                            state[4, ny, nx] = False
        # ----------------------------------

        # Place Key
        self.key_cell = (self.rng.randint(1, env_height - 1), self.rng.randint(1, wall_column))
        clear_safety_bubble(*self.key_cell) # <--- Apply 3x3 Clear
        state[6, self.key_cell[0], self.key_cell[1]] = True
        state[5, self.key_cell[0], self.key_cell[1]] = False
        
        # Place Goal
        self.end_marker_cell = (self.rng.randint(1, env_height - 1), self.rng.randint(wall_column + 1, env_width - 1))
        clear_safety_bubble(*self.end_marker_cell) # <--- Apply 3x3 Clear
        state[6, self.end_marker_cell[0], self.end_marker_cell[1]] = True
        state[5, self.end_marker_cell[0], self.end_marker_cell[1]] = False
        
        # Place Markers Layer
        state[5, :, :] = True
        state[5, self.key_cell[0], self.key_cell[1]] = False
        state[5, self.end_marker_cell[0], self.end_marker_cell[1]] = False
        state[6, self.key_cell[0], self.key_cell[1]] = True
        state[6, self.end_marker_cell[0], self.end_marker_cell[1]] = True

        # 4. ROBOT SPAWN
        valid_loc = False
        while not valid_loc:
            y_agent = self.rng.randint(1, env_height - 1)
            x_agent = self.rng.randint(1, wall_column)
            if not state[4, y_agent, x_agent] and not state[6, y_agent, x_agent]:
                valid_loc = True
                state[1, y_agent, x_agent] = True
                clear_safety_bubble(y_agent, x_agent) # <--- Apply 3x3 Clear

        # 5. DEFINE DOOR
        self.door_cells = [(2, wall_column), (3, wall_column)]
        self.door_locked = True
        state[4, self.door_cells[0][0], self.door_cells[0][1]] = True
        state[4, self.door_cells[1][0], self.door_cells[1][1]] = True

        return KarelEnvironment(initial_state=state, **run_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.door_locked = True

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.
        num_markers = env.markers_grid.sum()
        
        if self.door_locked:
            # If markers increase (robot put a marker down), penalty
            if num_markers > 2:
                terminated = True
                reward = self.crash_penalty
            # Check if key has been picked up (Key cell is now empty)
            elif env.markers_grid[self.key_cell[0], self.key_cell[1]] == 0:
                self.door_locked = False
                # UNLOCK: Remove the wall at door locations
                for door_cell in self.door_cells:
                    env.state[4, door_cell[0], door_cell[1]] = False
                reward = 0.5
        else:
            # Door is open, check for goal
            if num_markers > 1:
                # Check if end marker has been topped off (Goal reached)
                # In this logic, reaching goal often means putting a marker ON it
                # or just standing there depending on specific variant. 
                # This code assumes "topped off" means count == 2.
                if env.markers_grid[self.end_marker_cell[0], self.end_marker_cell[1]] == 2:
                    terminated = True
                    reward = 1.0 # SUCCESS! (Updated from 0.5 to 1.0 for clarity)
                else:
                    terminated = True
                    reward = self.crash_penalty
            elif num_markers == 0:
                terminated = True
                reward = self.crash_penalty
        
        return terminated, reward