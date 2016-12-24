import numpy as np


class RewardFunction:
    """Describe a reward function as a linear combination of function in a given basis"""

    # basis should not be changed, but weight can be.
    # basis are (dim_approx x states) matrix
    # weights are (dim_approx) array
    def __init__(self, basis, weights):
        self.dim_approx = len(weights)
        self.basis = basis
        self.weights = weights
        self.reward = np.dot(weights, basis)

    def update(self, weights):
        # Performs the update of the reward for new computed weight
        self.reward = np.dot(weights, self.basis)

    def estimatedvalue(self, trajectories, gamma, x_0):
        """ Compute the estimated value as a linear combination of accumulated reward for a given trajectories
            Assuming the trajectories are simulated by a given policy, gives an estimation of the value of this policy
            Input : trajectories (nb_trajectories x length_trajectories) array
                    gamma, decay rate of value over time
                    x_0 starting state
        """
        V = np.array(
                [np.sum(np.array([gamma ** k * self.basis[i][state] for (k, state) in np.ndenumerate(trajectories)]))
                 for i in range(self.dim_approx)])
        V = np.dot(self.weights, V)
        return V
