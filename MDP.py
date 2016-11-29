from scipy.stats import rv_discrete
import numpy as np


class MDP:
    def __init__(self, states, actions, dynamics, rewards, gamma):
        self.states = states
        self.actions = actions
        # dynamics[state, action] is the list of probabilities for next_state
        assert dynamics.shape == (len(states), len(actions))
        self.dynamics = dynamics
        # rewards[state] is deterministic, as in the paper
        assert len(rewards) == len(states)
        self.rewards = rewards
        self.gamma = gamma

    def simulate(self, state, action):
        rand_var = rv_discrete(values=(self.states, self.dynamics[state, action]))
        next_state = rand_var.rvs()
        reward = self.rewards[next_state]
        return next_state, reward

    def bellman_operator(self, policy, values):
        result = self.rewards
        result += self.gamma * np.array([self.dynamics[x, policy[x]] for x in self.states]).dot(values)
        return result

    def q_function(self, policy):
        values = self.bellman_estimator(policy)
        Q = np.zeros(len(self.states), len(self.actions))
        Q += self.rewards
        for x in self.states:
            for a in self.actions:
                Q[x, a] += self.gamma * np.array([self.dynamics[x, a] for x in self.states]).dot(values)
        return Q

    def bellman_estimator(self, policy, thresh=0.01):
        values = np.random.rand(len(self.states))
        new_values = self.bellman_operator(values, policy)
        while np.max(abs(new_values - values)) > thresh:
            values = new_values
            new_values = self.bellman_operator(values, policy)
        return new_values
