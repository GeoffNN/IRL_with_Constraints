"""Here we define the MDPs on which we test our algorithms."""
from MDP import MDP
import numpy as np
from policy import Policy
import seaborn as sb


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
        super().__init__(states, actions, dynamics, rewards, [self.n_squares-1], gamma)




def plot_Easy_Maze_trajectory(maze, trajectory):
    agg = np.zeros((len(maze.states), len(maze.states)))
    for i in maze.states:
        agg[np.unravel_index(i, (maze.side, maze.side))] = np.sum(trajectory[trajectory == i])
    sb.heatmap(agg)


Easy_Maze = EasyMaze()

simple_pol_arr = np.zeros(len(Easy_Maze.states))
for x in Easy_Maze.states:
    j, k = np.unravel_index(x, (Easy_Maze.side, Easy_Maze.side))
    if j >= k:
        simple_pol_arr[x] = Easy_Maze.right
    else:
        simple_pol_arr[x] = Easy_Maze.down

simple_pol = Policy(simple_pol_arr)


Easy_Maze.simulate_next_state(19, Easy_Maze.down)
