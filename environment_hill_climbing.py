"""Hill climb over map layouts to find environments solvable by a fixed DSL program.

Usage:
    python environment_hill_climbing.py cleanhouse.dsl 1 5000
    python environment_hill_climbing.py cleanhouse.dsl [n_successes] [n_iterations] [k] [n_toggles]

    k — how many mutated maps to try each iteration before giving up. If none of the k candidates improve the reward, it does a random restart.                                                                                                                                                                                                                                                                            
    n_toggles — how many wall cells to flip per mutation. 1 means change a single wall to floor (or floor to wall). Higher = bigger jumps in the search space.
    Default: 10, 1000, 8, 3
"""

import sys
import os
import copy
import json
import numpy as np

import prog_policies.utils
from prog_policies.karel_tasks.clean_house import CleanHouse
from prog_policies.karel.dsl import KarelDSL

ENV_HEIGHT = 14
ENV_WIDTH = 22
AGENT_POS = (1, 13)
AGENT_FACING = 2  # South
MAX_SIM_STEPS = 1500
NUM_MARKERS = 10


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_layout(layout, program_node, seed=0):
    """Run the program on a map built from `layout`. Returns (total_reward, visited_cells).

    Runs exactly like test_original.py: directly on the task's env, accumulating reward.
    """
    env_args = {'env_height': ENV_HEIGHT, 'env_width': ENV_WIDTH, 'layout': layout}
    task = CleanHouse(seed=seed, env_args=env_args)
    env = task.initial_environment

    step_gen = program_node.run_generator(env)

    visited = []
    if hasattr(env, 'hero_pos'):
        visited.append(tuple(env.hero_pos))

    steps = 0
    terminated = False
    total_reward = 0.0

    try:
        for _ in step_gen:
            steps += 1
            if steps > MAX_SIM_STEPS:
                break
            if hasattr(env, 'hero_pos'):
                visited.append(tuple(env.hero_pos))
            terminated, reward = task.get_reward(env)
            total_reward += reward
            if terminated:
                break

        if not terminated:
            terminated, reward = task.get_reward(env)
            total_reward += reward
    except Exception:
        total_reward = task.crash_penalty

    return total_reward, visited


def map_to_string(state):
    _, h, w = state.shape
    out = ""
    for y in range(h):
        row = ""
        for x in range(w):
            if state[4, y, x]:
                row += "#"
            elif np.any(state[0:4, y, x]):
                row += "A"
            elif state[6, y, x]:
                row += "*"
            else:
                row += "."
        out += row + "\n"
    return out


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

DEFAULT_LAYOUT = [
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-',   0,   0, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',   0, '-',   0,   0,   0,   0,   0,   0, '-'],
    ['-',   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-', '-', '-',   0, '-', '-'],
    ['-', '-',   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
    ['-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
    ['-', '-', '-',   0, '-',   0, '-', '-', '-',   0, '-',   0,   0, '-', '-', '-',   0, '-',   0, '-', '-', '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0, '-', '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0, '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-', '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
    ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
]


def get_interior_cells(layout):
    """Return set of (r, c) that are interior (not on the border)."""
    cells = set()
    for r in range(1, ENV_HEIGHT - 1):
        for c in range(1, ENV_WIDTH - 1):
            cells.add((r, c))
    return cells


def get_protected_cells():
    """Cells that must never become walls (agent start + neighbours needed for spawning)."""
    protected = {AGENT_POS}
    r, c = AGENT_POS
    # The marker near start at (r+1, c-1) and the agent cell itself
    protected.add((r + 1, c - 1))
    return protected


def layout_to_wall_set(layout):
    """Extract the set of (r,c) that are walls in the interior."""
    walls = set()
    for r in range(1, ENV_HEIGHT - 1):
        for c in range(1, ENV_WIDTH - 1):
            if layout[r][c] == '-':
                walls.add((r, c))
    return walls


def wall_set_to_layout(walls):
    """Build a full layout from a set of interior wall positions."""
    layout = []
    for r in range(ENV_HEIGHT):
        row = []
        for c in range(ENV_WIDTH):
            if r == 0 or r == ENV_HEIGHT - 1 or c == 0 or c == ENV_WIDTH - 1:
                row.append('-')
            elif (r, c) in walls:
                row.append('-')
            else:
                row.append(0)
        layout.append(row)
    return layout


def flood_fill_reachable(layout, start):
    """BFS from start, return set of reachable floor cells."""
    reachable = set()
    queue = [start]
    reachable.add(start)
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ENV_HEIGHT and 0 <= nc < ENV_WIDTH and (nr, nc) not in reachable:
                if layout[nr][nc] != '-':
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
    return reachable


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate_layout(layout, rng, n_toggles=3, protected=None):
    """Mutate a layout by toggling n_toggles interior cells between wall/floor.

    Ensures:
    - Protected cells are never turned into walls
    - Agent start remains reachable (connected floor)
    """
    if protected is None:
        protected = get_protected_cells()

    walls = layout_to_wall_set(layout)
    interior = get_interior_cells(layout)
    mutable = list(interior - protected)

    for _ in range(n_toggles):
        cell = mutable[rng.randint(len(mutable))]
        if cell in walls:
            walls.discard(cell)
        else:
            walls.add(cell)

    new_layout = wall_set_to_layout(walls)

    # Connectivity check: agent start must reach at least some floor cells
    reachable = flood_fill_reachable(new_layout, AGENT_POS)
    if len(reachable) < 10:
        # Mutation made map too closed off — return original
        return copy.deepcopy(layout)

    return new_layout


# ---------------------------------------------------------------------------
# Hill climbing
# ---------------------------------------------------------------------------

def hill_climb(program_node, n_iterations=1000, k=8, n_toggles=3,
               seed=42, n_successes_target=10, output_dir="hc_results"):
    """Hill climb over map layouts to find environments solvable by program_node.

    Args:
        program_node: Parsed DSL program
        n_iterations: Max iterations
        k: Number of neighbor candidates per iteration
        n_toggles: Number of wall cells to toggle per mutation
        seed: RNG seed
        n_successes_target: Stop after finding this many solvable maps
        output_dir: Directory to save results
    """
    rng = np.random.RandomState(seed)
    protected = get_protected_cells()

    os.makedirs(output_dir, exist_ok=True)

    # First: verify the program works on the original map with seed=0
    best_layout = copy.deepcopy(DEFAULT_LAYOUT)
    best_reward, best_visited = evaluate_layout(best_layout, program_node, seed=0)
    print(f"[Init] Original map with seed=0: reward={best_reward:.4f}")
    if best_reward >= 1.0:
        print("[Init] CONFIRMED: Program solves the original map.")
    else:
        print(f"[Init] WARNING: Program does NOT solve the original map (reward={best_reward:.4f}).")
        print(f"[Init] Visited {len(set(best_visited))} unique cells.")

    successes = []

    for iteration in range(1, n_iterations + 1):
        if best_reward >= 1.0:
            print(f"\n[SUCCESS #{len(successes)+1}] Iteration {iteration}, reward={best_reward:.4f}")
            successes.append(copy.deepcopy(best_layout))

            # Save this solution
            _save_layout(best_layout, program_node, len(successes), output_dir, rng)

            if len(successes) >= n_successes_target:
                print(f"\nReached target of {n_successes_target} solvable maps. Done.")
                break

            # Restart from a mutation of the successful layout to find diverse maps
            best_layout = mutate_layout(best_layout, rng, n_toggles=n_toggles * 3, protected=protected)
            best_reward, best_visited = evaluate_layout(best_layout, program_node, seed=rng.randint(2**31))
            print(f"[Restart] Mutated from success, new reward: {best_reward:.4f}")
            continue

        # Generate k neighbors, accept first improvement
        improved = False
        for _ in range(k):
            candidate = mutate_layout(best_layout, rng, n_toggles=n_toggles, protected=protected)
            # Evaluate with a random seed for marker placement diversity
            r, v = evaluate_layout(candidate, program_node, seed=rng.randint(2**31))
            if r > best_reward:
                best_layout = candidate
                best_reward = r
                best_visited = v
                improved = True
                break

        if iteration % 50 == 0 or improved:
            print(f"[Iter {iteration:4d}] reward={best_reward:.4f}  successes={len(successes)}  {'improved' if improved else ''}")

        # If stuck in local maximum, do a random restart with larger perturbation
        if not improved:
            # Random restart with bigger mutation
            best_layout = mutate_layout(best_layout, rng, n_toggles=n_toggles * 5, protected=protected)
            best_reward, best_visited = evaluate_layout(best_layout, program_node, seed=rng.randint(2**31))

    print(f"\nFinished. Found {len(successes)} solvable maps in {iteration} iterations.")
    return successes


def _save_layout(layout, program_node, index, output_dir, rng):
    """Save a successful layout to disk with its visualization."""
    env_args = {'env_height': ENV_HEIGHT, 'env_width': ENV_WIDTH, 'layout': layout}
    task = CleanHouse(seed=rng.randint(2**31), env_args=env_args)
    env = task.initial_environment

    map_str = map_to_string(env.state)
    filepath = os.path.join(output_dir, f"map_{index:04d}.txt")
    with open(filepath, 'w') as f:
        f.write(map_str)

    layout_path = os.path.join(output_dir, f"layout_{index:04d}.json")
    with open(layout_path, 'w') as f:
        json.dump(layout, f)

    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python environment_hill_climbing.py <solution.dsl> [n_successes] [n_iterations] [k] [n_toggles]")
        return

    dsl_path = sys.argv[1]
    n_successes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    n_toggles = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    with open(dsl_path, 'r') as f:
        dsl_code = f.read().strip()

    print(f"[Input] DSL: {dsl_path}")
    print(f"[Config] targets={n_successes}, max_iter={n_iterations}, k={k}, toggles={n_toggles}")

    dsl_parser = KarelDSL()
    try:
        program_node = dsl_parser.parse_str_to_node(dsl_code)
    except Exception as e:
        print(f"DSL Parse Error: {e}")
        return

    successes = hill_climb(
        program_node,
        n_iterations=n_iterations,
        k=k,
        n_toggles=n_toggles,
        n_successes_target=n_successes,
    )

    print(f"\nTotal solvable maps found: {len(successes)}")


if __name__ == "__main__":
    main()
