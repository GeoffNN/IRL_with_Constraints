import numpy as np

from problems import EasyMaze


def test_easy_maze_dynamics():
    # Tests that dynamics probabilities sums to 1 for each action
    easy_maze = EasyMaze()
    for a in easy_maze.actions:
        assert (easy_maze.dynamics[:, a].sum(axis=1) == np.ones(easy_maze.nb_states)).sum() == easy_maze.nb_states
