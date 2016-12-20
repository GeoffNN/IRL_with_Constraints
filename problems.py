"""Here we define the MDPs on which we test our algorithms."""
from MDP import MDP
import numpy as np


def easy_maze_init(side=5, gamma=.9):
    n_squares = side ** 2
    states = np.array(range(n_squares))
    left = 0
    right = 1
    up = 2
    down = 3
    actions = [left, right, up, down]
    n_actions = len(actions)
    dynamics = np.zeros((n_squares, n_actions))
    for k in range(n_squares):
        dynamics[k, up] = np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (-1, 0), (side, side),
                                               'clip')
        dynamics[k, down] = np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (1, 0), (side, side),
                                                 'clip')
        dynamics[k, right] = np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (0, 1), (side, side),
                                                  'clip')
        dynamics[k, left] = np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (0, -1), (side, side),
                                                 'clip')
    # Single reward in last square
    rewards = np.zeros(n_squares)
    rewards[n_squares - 1] = 100

    return states, actions, dynamics, rewards, gamma


Easy_Maze = MDP(*easy_maze_init())
