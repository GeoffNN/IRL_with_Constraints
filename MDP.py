import numpy as np
from scipy.stats import rv_discrete
from policy import Policy


class MDP:
    def __init__(self, states, actions, dynamics, rewards, term_states, gamma):
        self.states = states
        self.actions = actions
        # dynamics[state, action] is the list of probabilities for next_state
        assert dynamics.shape == (len(states), len(actions), len(states))
        self.dynamics = dynamics
        # rewards[state] is deterministic, as in the paper
        assert len(rewards) == len(states)
        self.rewards = rewards
        self.term_states = term_states
        self.gamma = gamma

    def simulate(self, x_0, policy, n_trajectories=100, trajectory_length=1000):
        trajectories = np.zeros((n_trajectories, trajectory_length))
        rewards = np.zeros((n_trajectories, trajectory_length))
        state = x_0
        for k in range(n_trajectories):
            for j in range(trajectory_length):
                trajectories[k, j] = state
                action = policy.next_action(state)
                next_state, rew = self.simulate_next_state(state, action)
                rewards[k, j] = rew
                state = next_state
        return trajectories, rewards

    def simulate_next_state(self, state, action):
        if state in self.term_states:
            return state, 0
        rand_var = rv_discrete(values=(self.states, self.dynamics[state, action]))
        next_state = rand_var.rvs()
        reward = self.rewards[next_state]

        return next_state, reward

    def bellman_operator(self, policy, values):
        result = self.rewards
        result += self.gamma * np.array([self.dynamics[x, policy.next_action(x)] for x in self.states]).dot(values)
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
        """Computes the values of a given policy"""
        values = np.random.rand(len(self.states))
        new_values = self.bellman_operator(policy, values)
        while np.max(abs(new_values - values)) > thresh:
            values = new_values
            new_values = self.bellman_operator(policy, values)
        return new_values

