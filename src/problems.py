"""Here we define the MDPs on which we test our algorithms."""
import numpy as np
import seaborn as sb

from src.MDP import MDP


class EasyMaze(MDP):
    def __init__(self, side=5, gamma=.9):
        self.n_squares = side ** 2
        self.side = side
        states = np.array(range(self.n_squares))
        self.left = 0
        self.right = 1
        self.up = 2
        self.down = 3
        actions = [self.left, self.right, self.up, self.down]
        n_actions = len(actions)
        dynamics = np.zeros((self.n_squares, n_actions, self.n_squares))
        for k in range(self.n_squares):
            dynamics[k, self.up] = [
                1 if j == np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (-1, 0), (side, side),
                                               'clip') else 0 for j in states]
            dynamics[k, self.down] = [
                1 if j == np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (1, 0), (side, side),
                                               'clip') else 0 for j in states]
            dynamics[k, self.right] = [
                1 if j == np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (0, 1), (side, side),
                                               'clip') else 0 for j in states]
            dynamics[k, self.left] = [
                1 if j == np.ravel_multi_index(np.array(np.unravel_index(k, (side, side))) + (0, -1), (side, side),
                                               'clip') else 0 for j in states]
        # Single reward in last square
        rewards = np.zeros(self.n_squares)
        rewards[self.n_squares - 1] = 100
        super().__init__(states, actions, dynamics, rewards, [self.n_squares - 1], gamma)

    def plot_trajectory(self, trajectory):
        agg = np.zeros((self.side, self.side))
        for i in self.states:
            if i not in self.term_states:
                agg[np.unravel_index(i, (self.side, self.side))] = np.sum(trajectory == i)
            else:
                agg[np.unravel_index(i, (self.side, self.side))] = 1
        sb.heatmap(agg)

    def plot_reward(self):
        agg = np.zeros((self.side, self.side))
        for i in self.states:
            agg[np.unravel_index(i, (self.side, self.side))] = self.rewards[i]
        sb.heatmap(agg)

