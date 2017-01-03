from numpy import allclose

from problems import GridWorld


def test_gridworld_dynamics():
    """Tests that dynamics probabilities sums to 1 for each action"""
    gridworld = GridWorld()
    for a in gridworld.actions:
        assert allclose(gridworld.dynamics[:, a, :].sum(axis=1), 1)
