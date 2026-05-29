from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.world_object import Wall
import os

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

# direction integer -> (col_delta, row_delta)
# RIGHT=0, DOWN=1, LEFT=2, UP=3
DIR_TO_VEC = {
    0: ( 1,  0),
    1: ( 0,  1),
    2: (-1,  0),
    3: ( 0, -1),
}

# direction vector -> integer
VEC_TO_DIR = {
    ( 1,  0): 0,
    ( 0,  1): 1,
    (-1,  0): 2,
    ( 0, -1): 3,
}

def parse_env(env):
    """
    Extract static map info from a loaded env.
    Returns walls, key/door/goal positions, and agent's initial state.
    """
    u      = env.unwrapped
    width  = u.width
    height = u.height

    walls          = set()
    key_pos        = None
    door_pos       = []
    init_door_open = []
    goal_pos       = None

    for row in range(height):
        for col in range(width):
            cell = u.grid.get(col, row)
            if isinstance(cell, Wall):
                walls.add((col, row))
            elif isinstance(cell, Key):
                key_pos = (col, row)
            elif isinstance(cell, Door):
                door_pos.append((col, row))
                init_door_open.append(cell.is_open)
            elif isinstance(cell, Goal):
                goal_pos = (col, row)

    # Convert direction vector e.g. [1,0] -> integer 0
    dir_vec  = tuple(int(v) for v in u.dir_vec)
    init_dir = VEC_TO_DIR[dir_vec]
    init_pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))

    return {
        "width":          width,
        "height":         height,
        "walls":          frozenset(walls),
        "key_pos":        key_pos,
        "door_pos":       door_pos,
        "goal_pos":       goal_pos,
        "init_pos":       init_pos,
        "init_dir":       init_dir,
        "init_door_open": tuple(init_door_open),
    }

def transition(state, action, walls, key_pos, door_pos):
    """
    Deterministic transition function for the Door-Key MDP.

    state = (col, row, direction, has_key, door_open)
        direction : int 0-3
        door_open : tuple of bool, one per door in door_pos order

    Invalid actions are no-ops: agent stays, cost still applies.
    Returns next_state.
    """
    col, row, direction, has_key, door_open = state
    door_open = list(door_open)

    # Turns only change direction
    if action == TL:
        return (col, row, (direction - 1) % 4, has_key, tuple(door_open))
    if action == TR:
        return (col, row, (direction + 1) % 4, has_key, tuple(door_open))

    # Compute the cell directly in front of the agent
    dc, dr = DIR_TO_VEC[direction]
    fc, fr = col + dc, row + dr

    if action == MF:
        if (fc, fr) in walls:                      # blocked by wall
            return state
        if (fc, fr) in door_pos:
            di = door_pos.index((fc, fr))
            if not door_open[di]:                  # blocked by closed door
                return state
        return (fc, fr, direction, has_key, tuple(door_open))

    if action == PK:
        # Key must be directly in front, agent must not already have it
        if (not has_key) and (key_pos is not None) and ((fc, fr) == key_pos):
            return (col, row, direction, True, tuple(door_open))
        return state  # no-op

    if action == UD:
        # Door must be directly in front, agent must have key
        if (fc, fr) in door_pos:
            di = door_pos.index((fc, fr))
            if has_key and not door_open[di]:
                door_open[di] = True
                return (col, row, direction, has_key, tuple(door_open))
        return state  # no-op
    
def dp_solve(env):
    """
    Solve the Door-Key MDP using value iteration.

    State: (col, row, direction, has_key, door_open)
    Terminal state: agent reaches goal_pos -> cost-to-go = 0
    Invalid actions are no-ops: agent stays in place, cost still applies.
    All other states initialized to infinity, iterated until convergence.
    """
    m         = parse_env(env)
    width     = m["width"]
    height    = m["height"]
    walls     = m["walls"]
    key_pos   = m["key_pos"]
    door_pos  = m["door_pos"]
    goal_pos  = m["goal_pos"]
    num_doors = len(door_pos)

    # Build the full state space
    all_states = [
        (col, row, d, hk, do)
        for col in range(width)
        for row in range(height)
        for d   in range(4)
        for hk  in (False, True)
        for do  in (
            tuple(bool((bits >> i) & 1) for i in range(num_doors))
            for bits in range(2 ** num_doors)
        )
    ]

    # Initialize value function 
    V      = {}
    policy = {}
    for s in all_states:
        col, row = s[0], s[1]
        if (col, row) == goal_pos:
            V[s] = 0.0        # terminal state: zero cost-to-go
        else:
            V[s] = float('inf')

    # Value iteration
    for iteration in range(10000):
        delta = 0.0
        new_V = {}

        for s in all_states:
            col, row, direction, has_key, door_open = s

            # Terminal and wall states don't change
            if (col, row) == goal_pos:
                new_V[s] = 0.0
                continue
            if (col, row) in walls:
                new_V[s] = float('inf')
                continue

            # Key is on the floor only if agent doesn't have it
            kp = key_pos if not has_key else None

            best_val = float('inf')
            best_act = None
            for a in [MF, TL, TR, PK, UD]:
                s_next = transition(s, a, walls, kp, door_pos)
                cost   = step_cost(a) + V[s_next]
                if cost < best_val:
                    best_val = cost
                    best_act = a

            new_V[s]  = best_val
            policy[s] = best_act

            # Track max change for convergence check
            old = V[s]
            new = new_V[s]
            if old == float('inf') and new == float('inf'):
                pass  # both unreachable, no change
            elif old == float('inf') or new == float('inf'):
                delta = float('inf')  # state just became reachable
            else:
                delta = max(delta, abs(new - old))

        V = new_V
        if delta < 1e-6:
            print(f"  Converged in {iteration + 1} iterations")
            break

    return V, policy, m

def extract_action_seq(policy, m):
    """
    Simulate the policy from the initial state and collect actions.
    """
    state = (
        m["init_pos"][0],
        m["init_pos"][1],
        m["init_dir"],
        False,
        m["init_door_open"],
    )

    goal_pos = m["goal_pos"]
    walls    = m["walls"]
    door_pos = m["door_pos"]
    key_pos  = m["key_pos"]
    seq      = []

    for _ in range(1000):
        if (state[0], state[1]) == goal_pos:
            break
        a = policy[state]
        seq.append(a)
        kp    = key_pos if not state[3] else None
        state = transition(state, a, walls, kp, door_pos)

    return seq

# Fixed geometry for all 36 random 10x10 environments
RANDOM_DOOR_POS  = [(5, 3), (5, 7)]
RANDOM_KEY_LOCS  = [(2, 2), (2, 3), (1, 6)]
RANDOM_GOAL_LOCS = [(6, 1), (7, 3), (6, 6)]
RANDOM_INIT_POS  = (4, 8)
RANDOM_INIT_DIR  = 3  # UP

def build_random_walls():
    """
    Build the fixed wall layout for the random 10x10 environments.
    Perimeter walls + vertical wall at col 5, minus the two door cells.
    """
    walls = set()
    for c in range(10):
        walls.add((c, 0))
        walls.add((c, 9))
    for r in range(10):
        walls.add((0, r))
        walls.add((9, r))
    for r in range(10):
        walls.add((5, r))
    # Door cells are not walls
    walls -= set(RANDOM_DOOR_POS)
    return frozenset(walls)


def dp_solve_random():
    """
    Solve a single unified MDP covering all 36 random environments.

    Extended state: (col, row, direction, has_key, door_open, key_idx, goal_idx)
    """
    walls = build_random_walls()

    # Build extended state space
    all_states = [
        (col, row, d, hk, do, ki, gi)
        for col in range(10)
        for row in range(10)
        for d   in range(4)
        for hk  in (False, True)
        for do  in ((False,False),(False,True),(True,False),(True,True))
        for ki  in range(3)
        for gi  in range(3)
    ]
    print(f"  Total states: {len(all_states)}")

    # Initialize value function
    V      = {}
    policy = {}
    for s in all_states:
        col, row, d, hk, do, ki, gi = s
        if (col, row) == RANDOM_GOAL_LOCS[gi]:
            V[s] = 0.0
        else:
            V[s] = float('inf')

    # Value iteration
    for iteration in range(10000):
        delta = 0.0
        new_V = {}

        for s in all_states:
            col, row, d, hk, do, ki, gi = s

            if (col, row) == RANDOM_GOAL_LOCS[gi]:
                new_V[s] = 0.0
                continue
            if (col, row) in walls:
                new_V[s] = float('inf')
                continue

            kp = RANDOM_KEY_LOCS[ki] if not hk else None

            best_val = float('inf')
            best_act = None
            for a in [MF, TL, TR, PK, UD]:
                s_next = transition_random(s, a, walls, kp)
                cost   = step_cost(a) + V[s_next]
                if cost < best_val:
                    best_val = cost
                    best_act = a

            new_V[s]  = best_val
            policy[s] = best_act

            old = V[s]
            new = new_V[s]
            if old == float('inf') and new == float('inf'):
                pass
            elif old == float('inf') or new == float('inf'):
                delta = float('inf')
            else:
                delta = max(delta, abs(new - old))

        V = new_V
        if delta < 1e-6:
            print(f"  Converged in {iteration + 1} iterations")
            break

    return V, policy, walls


def transition_random(state, action, walls, key_pos):
    """
    Transition function for the extended random-map state.
    Same logic as transition() but uses fixed RANDOM_DOOR_POS.
    """
    col, row, d, hk, do, ki, gi = state
    do = list(do)

    if action == TL:
        return (col, row, (d-1)%4, hk, tuple(do), ki, gi)
    if action == TR:
        return (col, row, (d+1)%4, hk, tuple(do), ki, gi)

    dc, dr = DIR_TO_VEC[d]
    fc, fr = col + dc, row + dr

    if action == MF:
        if (fc, fr) in walls:
            return state
        if (fc, fr) in RANDOM_DOOR_POS:
            di = RANDOM_DOOR_POS.index((fc, fr))
            if not do[di]:
                return state
        return (fc, fr, d, hk, tuple(do), ki, gi)

    if action == PK:
        if (not hk) and (key_pos is not None) and ((fc, fr) == key_pos):
            return (col, row, d, True, tuple(do), ki, gi)
        return state

    if action == UD:
        if (fc, fr) in RANDOM_DOOR_POS:
            di = RANDOM_DOOR_POS.index((fc, fr))
            if hk and not do[di]:
                do[di] = True
                return (col, row, d, hk, tuple(do), ki, gi)
        return state

def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Template:
        Replace the placeholder list with the action sequence returned by your
        planner. Minimize the same total stage cost as in utils.step_cost (and
        as defined in your report's MDP). You may branch on env / loaded map if
        needed for Part (A); Part (B) should respect the single-policy requirement.
    """
    # STUDENT: placeholder sequence for wiring; not a solution for all maps.
    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    V, policy, m = dp_solve(env)
    return extract_action_seq(policy, m)


# def partA():
#     env_path = "./envs/known_envs/example-8x8.env"
#     env, info = load_env(env_path)  # load an environment
#     seq = doorkey_problem(env)  # find the optimal action sequence
#     draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save

def partA():
    maps = [
        "doorkey-5x5-normal",
        "doorkey-6x6-normal",
        "doorkey-6x6-direct",
        "doorkey-6x6-shortcut",
        "doorkey-8x8-normal",
        "doorkey-8x8-direct",
        "doorkey-8x8-shortcut",
    ]
    names = {0:'MF', 1:'TL', 2:'TR', 3:'PK', 4:'UD'}
    os.makedirs("./trajectories", exist_ok=True)

    for map_name in maps:
        env_path  = f"./envs/known_envs/{map_name}.env"
        env, info = load_env(env_path)
        seq       = doorkey_problem(env)
        cost      = sum(step_cost(a) for a in seq)
        print(f"{map_name}: cost={cost}, steps={len(seq)}")
        print(f"  {' -> '.join(names[a] for a in seq)}")

        # GIF
        env_gif, _ = load_env(env_path)
        draw_gif_from_seq(seq, env_gif, path=f"./gif/{map_name}.gif")

        # Trajectory image
        env_traj, _ = load_env(env_path)
        draw_trajectory(seq, env_traj,
                        path=f"./trajectories/{map_name}_trajectory.png")
        
# def partB():
#     env_folder = "./envs/random_envs"
#     env, info, env_path = load_random_env(env_folder)

def partB():
    print("\n--- Part B: Random Maps ---")
    os.makedirs("./trajectories", exist_ok=True)
    V, policy, walls = dp_solve_random()

    env_folder = "./envs/random_envs"
    env_files  = sorted(f for f in os.listdir(env_folder) if f.endswith(".env"))

    names = {0:'MF', 1:'TL', 2:'TR', 3:'PK', 4:'UD'}
    total_ok = 0

    for env_file in env_files:
        env_path = os.path.join(env_folder, env_file)
        with open(env_path, "rb") as f:
            env = pickle.load(f)

        # Find key, goal, and door states in this specific env
        key_pos   = None
        goal_pos  = None
        door_open = [False, False]

        for row in range(10):
            for col in range(10):
                cell = env.unwrapped.grid.get(col, row)
                if isinstance(cell, Key):
                    key_pos = (col, row)
                elif isinstance(cell, Goal):
                    goal_pos = (col, row)
                elif isinstance(cell, Door):
                    if (col, row) in RANDOM_DOOR_POS:
                        di = RANDOM_DOOR_POS.index((col, row))
                        door_open[di] = cell.is_open

        ki = RANDOM_KEY_LOCS.index(key_pos)
        gi = RANDOM_GOAL_LOCS.index(goal_pos)

        # Build initial extended state
        init_state = (
            RANDOM_INIT_POS[0], RANDOM_INIT_POS[1],
            RANDOM_INIT_DIR,
            False,
            tuple(door_open),
            ki, gi
        )

        # Simulate policy
        state = init_state
        seq   = []
        for _ in range(1000):
            col, row = state[0], state[1]
            if (col, row) == RANDOM_GOAL_LOCS[gi]:
                break
            a = policy[state]
            seq.append(a)
            kp    = RANDOM_KEY_LOCS[ki] if not state[3] else None
            state = transition_random(state, a, walls, kp)

        cost    = sum(step_cost(a) for a in seq)
        reached = (state[0], state[1]) == RANDOM_GOAL_LOCS[gi]
        status  = "OK" if reached else "FAIL"
        if reached:
            total_ok += 1

        print(f"  [{status}] {env_file}: cost={cost}, steps={len(seq)}")
        draw_gif_from_seq(seq, env, path=f"./gif/{env_file.replace('.env','.gif')}")

        # Trajectory image
        with open(env_path, "rb") as f:
            env_traj = pickle.load(f)
        draw_trajectory(seq, env_traj,
                        path=f"./trajectories/{env_file.replace('.env','_trajectory.png')}")

    print(f"\n  Solved {total_ok}/{len(env_files)} environments")

if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    partB()

