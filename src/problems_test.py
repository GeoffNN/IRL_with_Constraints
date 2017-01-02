import numpy as np

from problems import EasyMaze


def test_easy_maze():
    easy_maze = EasyMaze()
    for a in easy_maze.actions:
        assert easy_maze.dynamics[:, a].sum(axis=1) == np.ones(easy_maze.side)
