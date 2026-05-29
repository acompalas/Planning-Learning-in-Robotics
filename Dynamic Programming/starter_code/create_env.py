import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import pickle
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key, Door
import numpy as np

WALL = 0
FLOOR = 1
KEY = 2
AGENT = 3
DOOR = 4
GOAL = 5

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

known_map_configs = {
    "doorkey-5x5-normal": {
        "size": 5,
        "agent_init_direction": DOWN,
        "layout": np.array([[FLOOR, KEY, WALL], [AGENT, DOOR, FLOOR], [FLOOR, WALL, GOAL]]),
    },
    "doorkey-6x6-direct": {
        "size": 6,
        "agent_init_direction": RIGHT,
        "layout": np.array(
            [
                [FLOOR, AGENT, FLOOR, FLOOR],
                [KEY, FLOOR, WALL, FLOOR],
                [FLOOR, WALL, WALL, GOAL],
                [FLOOR, DOOR, FLOOR, FLOOR],
            ]
        ),
    },
    "doorkey-6x6-normal": {
        "size": 6,
        "agent_init_direction": LEFT,
        "layout": np.array(
            [
                [FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, AGENT, DOOR, FLOOR],
                [FLOOR, FLOOR, WALL, GOAL],
                [KEY, FLOOR, WALL, FLOOR],
            ]
        ),
    },
    "doorkey-6x6-shortcut": {
        "size": 6,
        "agent_init_direction": LEFT,
        "layout": np.array(
            [
                [FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, WALL, FLOOR],
                [KEY, AGENT, DOOR, GOAL],
                [FLOOR, FLOOR, WALL, FLOOR],
            ]
        ),
    },
    "doorkey-8x8-direct": {
        "size": 8,
        "agent_init_direction": DOWN,
        "layout": np.array(
            [
                [FLOOR, AGENT, WALL, FLOOR, GOAL, FLOOR],
                [FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR],
                [WALL, WALL, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, KEY, DOOR, FLOOR, FLOOR],
            ]
        ),
    },
    "doorkey-8x8-normal": {
        "size": 8,
        "agent_init_direction": RIGHT,
        "layout": np.array(
            [
                [FLOOR, AGENT, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, DOOR, FLOOR, FLOOR],
                [WALL, WALL, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, WALL, FLOOR, GOAL],
                [FLOOR, FLOOR, KEY, WALL, FLOOR, FLOOR],
            ]
        ),
    },
    "doorkey-8x8-shortcut": {
        "size": 8,
        "agent_init_direction": UP,
        "layout": np.array(
            [
                [FLOOR, AGENT, FLOOR, DOOR, GOAL, FLOOR],
                [FLOOR, FLOOR, KEY, WALL, FLOOR, FLOOR],
                [WALL, WALL, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, WALL, FLOOR, FLOOR],
                [FLOOR, WALL, WALL, WALL, FLOOR, FLOOR],
                [FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR],
            ]
        ),
    },
    "example-8x8": {
        "size": 8,
        "agent_init_direction": RIGHT,
        "layout": np.array(
            [
                [FLOOR, FLOOR, WALL, FLOOR, FLOOR, FLOOR],
                [FLOOR, FLOOR, WALL, FLOOR, FLOOR, FLOOR],
                [FLOOR, AGENT, WALL, FLOOR, FLOOR, FLOOR],
                [FLOOR, FLOOR, DOOR, FLOOR, FLOOR, FLOOR],
                [FLOOR, KEY, WALL, FLOOR, FLOOR, GOAL],
                [FLOOR, FLOOR, WALL, FLOOR, FLOOR, FLOOR],
            ]
        ),
    },
}


def create_known_envs(map_name):
    config = known_map_configs[map_name]
    size = config["size"]
    layout = config["layout"]

    env_wrapper = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0", render_mode="rgb_array")
    env_wrapper.reset()
    env: DoorKeyEnv = env_wrapper.env.env
    env.grid = Grid(size, size)
    env.grid.wall_rect(0, 0, size, size)

    for i in range(size - 2):
        ii = i + 1
        for j in range(size - 2):
            jj = j + 1
            if layout[i, j] == WALL:
                env.grid.set(jj, ii, Wall())  # .set(column, row, object)
            elif layout[i, j] == FLOOR:
                env.grid.set(jj, ii, None)
            elif layout[i, j] == KEY:
                env.grid.set(jj, ii, Key(color="yellow"))
            elif layout[i, j] == DOOR:
                env.grid.set(jj, ii, Door(color="yellow", is_locked=True))
            elif layout[i, j] == GOAL:
                env.grid.set(jj, ii, Goal())
            elif layout[i, j] == AGENT:
                env.agent_pos = (jj, ii)
                env.agent_dir = config["agent_init_direction"]
                env.grid.set(jj, ii, None)
            else:
                raise ValueError(f"Invalid layout code: {layout[i, j]}")
    env.gen_obs()
    image = env.render()
    plt.imsave(f"envs/known_envs/{map_name}.png", image)
    with open(f"envs/known_envs/{map_name}.env", "wb") as f:
        pickle.dump(env_wrapper, f)

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)


register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

def create_random_envs():
    size = 10
    key_locations = [(2, 2), (2, 3), (1, 6)] 
    goal_locations = [(6, 1), (7, 3), (6, 6)] 

    cnt = 0
    for key_jj, key_ii in key_locations:
        for goal_jj, goal_ii in goal_locations:
            for door1_opened in [True, False]:
                for door2_opened in [True, False]:
                    cnt += 1
                    env_wrapper = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0", render_mode="rgb_array")
                    env_wrapper.reset()
                    env: DoorKeyEnv = env_wrapper.env.env
                    env.grid = Grid(size, size)
                    env.grid.vert_wall(5, 0)

                    door1 = Door(color="yellow", is_open=door1_opened, is_locked=not door1_opened)
                    env.grid.set(5, 3, door1)
                    door2 = Door(color="yellow", is_open=door2_opened, is_locked=not door2_opened)
                    env.grid.set(5, 7, door2)
                    env.grid.set(key_jj, key_ii, Key(color="yellow"))
                    env.grid.set(goal_jj, goal_ii, Goal())
                    env.agent_pos = (4, 8)
                    env.agent_dir = UP
                    env.gen_obs()
                    image = env.render()
                    plt.imsave(f"envs/random_envs/DoorKey-{size}x{size}-{cnt}.png", image)
                    with open(f"envs/random_envs/DoorKey-{size}x{size}-{cnt}.env", "wb") as f:
                        pickle.dump(env_wrapper, f)


def main():
    for map_name in known_map_configs.keys():
        create_known_envs(map_name)
    create_random_envs()


if __name__ == "__main__":
    main()

