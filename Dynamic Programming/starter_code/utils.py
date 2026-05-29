import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def step_cost(action):
    """
    Template (stage cost): map each action index to a nonnegative scalar.

    - Keep this consistent with the stage cost l(x, u) in your report and DP
    - Non-uniform costs (e.g., different prices for turn vs. forward vs. unlock)
    - Invalid or no-op steps: decide in your formulation whether they incur cost
    """
    # STUDENT: placeholder only (uniform zero cost). Remove when implementing.
    if action in [TL, TR]:
        return 1
    if action == MF:
        return 3
    if action == PK:
        return 2
    if action == UD:
        return 5
    raise ValueError(f"Invalid action: {action}")


def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.unwrapped.actions.forward,
        1: env.unwrapped.actions.left,
        2: env.unwrapped.actions.right,
        3: env.unwrapped.actions.pickup,
        4: env.unwrapped.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return step_cost(action), terminated


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height, 
        "width": env.unwrapped.width, 
        "init_agent_pos": env.unwrapped.agent_pos, 
        "init_agent_dir": env.unwrapped.dir_vec,
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.unwrapped.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith(".env")]
    env_path = random.choice(env_list)
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, path="./gif/doorkey.gif"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            step(env, act)
            img = env.render()
            writer.append_data(img)
    print(f"GIF is written to {path}")
    return
    
def draw_trajectory(seq, env, path="./gif/trajectory.png"):
    """
    Save a static image showing the agent's trajectory as a red line
    overlaid on the final environment state.
    
    The red line connects the center of each cell visited in order.
    The agent is shown in its final position and orientation.
    """

    # Replay the sequence and record positions
    positions = [tuple(env.unwrapped.agent_pos)]
    for act in seq:
        step(env, act)
        positions.append(tuple(env.unwrapped.agent_pos))

    # Render the final frame
    img = env.render()
    img = img.astype(np.float32)

    height, width, _ = img.shape
    grid_h = env.unwrapped.height
    grid_w = env.unwrapped.width
    cell_h = height // grid_h
    cell_w = width  // grid_w

    def cell_center(col, row):
        # Convert grid (col, row) to pixel (x, y)
        px = int(col * cell_w + cell_w // 2)
        py = int(row * cell_h + cell_h // 2)
        return px, py

    # Draw red line between consecutive positions
    import cv2
    img_uint8 = img.astype(np.uint8)
    for i in range(len(positions) - 1):
        p1 = cell_center(positions[i][0],     positions[i][1])
        p2 = cell_center(positions[i+1][0],   positions[i+1][1])
        cv2.line(img_uint8, p1, p2, color=(255, 0, 0), thickness=2)

    # Draw a red dot at each visited cell
    for pos in positions:
        cx, cy = cell_center(pos[0], pos[1])
        cv2.circle(img_uint8, (cx, cy), radius=3,
                   color=(255, 0, 0), thickness=-1)

    plt.imsave(path, img_uint8)
    print(f"Trajectory image saved to {path}")