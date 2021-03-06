"""Here we define the MDPs on which we test our algorithms."""
import numpy as np
import seaborn as sb

from src.MDP import MDP
from src.reward import RewardFunction


class GridWorld(MDP):
    # TODO: Change dynamics for slight random behavior
    def __init__(self, side=5, gamma=.9, wind=.3):
        self.n_squares = side ** 2
        self.side = side
        self.wind = wind
        states = np.array(range(self.n_squares))
        self.left = 0
        self.right = 1
        self.up = 2
        self.down = 3
        actions = [self.left, self.right, self.up, self.down]
        n_actions = len(actions)
        dynamics = np.zeros((self.n_squares, n_actions, self.n_squares))
        # Set dynamics
        for k in range(self.n_squares):
            dynamics[k, self.up] = np.array([
                                                1 - self.wind if j == self.up_square(k) else 0 for j in states])
            dynamics[k, self.down] = np.array([
                                                  1 - self.wind if j == self.down_square(k) else 0 for j in states])
            dynamics[k, self.right] = np.array([
                                                   1 - self.wind if j == self.right_square(k) else 0 for j in states])
            dynamics[k, self.left] = np.array([
                                                  1 - self.wind if j == self.left_square(k) else 0 for j in states])
            neighbors = list(self.neighbors(k))
            for j in neighbors:
                dynamics[k, :, j] += self.wind / len(neighbors)

        # Reward for transition to the exit
        rewards = np.zeros((self.n_squares, self.n_squares))
        for j in self.neighbors(self.n_squares - 1):
            rewards[j, -1] = 1
        # TODO: set a reward of form R(x,y)
        reward_function = RewardFunction([1], [1])

        super().__init__(states, actions, dynamics, reward_function, rewards, [self.n_squares - 1], gamma)

    def neighbors(self, k):
        for j in self.up_square(k), self.down_square(k), self.right_square(k), self.left_square(k):
            if j != k:
                yield j

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
