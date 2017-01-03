import numpy as np
import seaborn as sb
from numpy.linalg import cholesky
from scipy.stats import norm

from problems import GridWorld
from structure_finding import compute_reward_nonneg, constraints_nonneg, objective_matrix_nonneg


def reward_tryout(mdp):
    """Gives a typical reward to tune"""
    rew = mdp.true_reward
    # Add noise
    rew += norm.rvs(0, 0.1, len(rew))
    return rew


def test_constraints_nonneg():
    easy_maze = GridWorld()
    rew = reward_tryout(easy_maze)
    G, h = constraints_nonneg(easy_maze, rew)
    assert np.all(np.array(G).sum(axis=1) == (1 - easy_maze.gamma) * np.ones(easy_maze.nb_states ** 2))
    assert G.size[0] == h.size[0]


def test_objective_mat_nonneg_test():
    easy_maze = GridWorld()
    rew = reward_tryout(easy_maze)
    Q = objective_matrix_nonneg(easy_maze, rew)
    assert np.array(Q).shape == (easy_maze.nb_states, easy_maze.nb_states)
    # Tests if Q is positive definite (cubic in Q.size)
    cholesky(Q)


def test_nonneg_result():
    easy_maze = GridWorld()
    rew = reward_tryout(easy_maze)
    var_rew = compute_reward_nonneg(easy_maze, rew)
    sb.heatmap(var_rew)
    assert np.all(var_rew < .5)