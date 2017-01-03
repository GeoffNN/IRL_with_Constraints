import numpy as np
from numpy.linalg import cholesky
from scipy.stats import norm

from problems import EasyMaze
from structure_finding import compute_reward_nonneg, constraints_nonneg, objective_matrix_nonneg


def reward_tryout(length):
    """Gives a typical reward to tune"""
    rew = np.zeros(length)
    rew[-1] = 1
    # Add noise
    rew += norm.rvs(0, 0.1, len(rew))
    return rew


def test_constraints_nonneg():
    easy_maze = EasyMaze()
    rew = reward_tryout(easy_maze.nb_states ** 2)
    G, h = constraints_nonneg(easy_maze, rew)
    assert np.all(np.array(G).sum(axis=1) == (1 - easy_maze.gamma) * np.ones(easy_maze.nb_states ** 2))
    assert G.size[0] == h.size[0]


def test_objective_mat_nonneg_test():
    easy_maze = EasyMaze()
    rew = reward_tryout(easy_maze.nb_states ** 2)
    Q = objective_matrix_nonneg(easy_maze, rew)
    assert np.array(Q).shape == (easy_maze.nb_states, easy_maze.nb_states)
    cholesky(Q)


def test_nonneg_result():
    easy_maze = EasyMaze()
    rew = reward_tryout(easy_maze.nb_states ** 2)
    reward = np.array(compute_reward_nonneg(easy_maze, rew))
    assert np.all(easy_maze.gamma * np.abs(np.array([[r1 - r2 for r1 in reward] for r2 in reward]) < .5))
