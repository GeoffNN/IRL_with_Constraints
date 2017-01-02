"""Here we define the MDPs on which we test our algorithms."""
import numpy as np
import seaborn as sb

from src.MDP import MDP
from src.reward import RewardFunction


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
                1 if j == self.up_square(k) else 0 for j in states]
            dynamics[k, self.down] = [
                1 if j == self.down_square(k) else 0 for j in states]
            dynamics[k, self.right] = [
                1 if j == self.right_square(k) else 0 for j in states]
            dynamics[k, self.left] = [
                1 if j == self.left_square(k) else 0 for j in states]
        # Single reward in last square
        rewards = np.zeros(self.n_squares)
        rewards[self.n_squares - 1] = 100

        # A changer c est bien de la merde la
        reward_function = RewardFunction([1], [1])

        super().__init__(states, actions, dynamics, reward_function, rewards, [self.n_squares - 1], gamma)

    def up_square(self, k):
        return np.ravel_multi_index(np.array(np.unravel_index(k, (self.side, self.side))) + (-1, 0),
                                    (self.side, self.side), 'clip')

    def down_square(self, k):
        return np.ravel_multi_index(np.array(np.unravel_index(k, (self.side, self.side))) + (1, 0),
                                    (self.side, self.side), 'clip')

    def right_square(self, k):
        return np.ravel_multi_index(np.array(np.unravel_index(k, (self.side, self.side))) + (0, 1),
                                    (self.side, self.side), 'clip')

    def left_square(self, k):
        return np.ravel_multi_index(np.array(np.unravel_index(k, (self.side, self.side))) + (0, -1),
                                    (self.side, self.side), 'clip')

    def plot_trajectory(self, trajectory):
        agg = np.zeros((self.side, self.side))
        for i in self.states:
            if i not in self.term_states:
                agg[np.unravel_index(i, (self.side, self.side))] = np.sum(trajectory == i)
            else:
                agg[np.unravel_index(i, (self.side, self.side))] = 1
        sb.heatmap(agg)

    def plot_true_reward(self):
        agg = np.zeros((self.side, self.side))
        for i in self.states:
            agg[np.unravel_index(i, (self.side, self.side))] = self.true_reward[i]
        sb.heatmap(agg)
