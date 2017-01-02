import numpy as np
from scipy.stats import rv_discrete

from src.policy import Policy


class MDP:
    def __init__(self, states, actions, dynamics, reward_function, true_reward, term_states, gamma):
        self.states = states
        self.nb_states = states.size
        self.actions = actions
        # dynamics[state, action] is the list of probabilities for next_state
        assert dynamics.shape == (len(states), len(actions), len(states))
        self.dynamics = dynamics
        # Important : reward_function contains three fields : basis, weights (mutable), and reward,
        # which correspond to a matrix such as reward[state] = rew
        # Meanwhile, true_reward is just a matrix giving the ground_truth. Must not be used EXCEPT
        # to compute the (supposed unknown) optimal policy of our problem and simulate a sample of
        # trajectories from that, or to compare the computed reward with the ground truth
        assert len(true_reward) == len(states)
        self.reward_function = reward_function
        self.true_reward = true_reward
        self.term_states = term_states
        assert 0 <= gamma <= 1
        self.gamma = gamma

    def simulate_next_state(self, state, action):
        """ For a given state,action, simulate the outcome (next state and reward obtained)
            With reward being the optimal policy
        """
        if state in self.term_states:
            return state, 0
        rand_var = rv_discrete(values=(self.states, self.dynamics[state, action]))
        next_state = rand_var.rvs()
        reward = self.true_reward[next_state]
        return next_state, reward

    def simulate(self, x_0, policy, n_trajectories=100, trajectory_length=1000):
        """
            Simulate 100 trajectories of length 1000 for a given policy and a starting state x_0
        """
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

    def bellman_operator(self, policy, values):
        result = self.reward_function.reward
        result += self.gamma * np.array([self.dynamics[x, policy.next_action(x)] for x in self.states]).dot(values)
        return result

    def bellman_estimator(self, policy, thresh=0.01):
        """Computes the values of a given policy for the current estimation of the reward"""
        values = np.random.rand(len(self.states))
        new_values = self.bellman_operator(policy, values)
        while np.max(abs(new_values - values)) > thresh:
            values = new_values
            new_values = self.bellman_operator(policy, values)
        return new_values

    def q_function(self, policy):
        """ Compute Q-function for current estimation of the reward """
        values = self.bellman_estimator(policy)
        Q = np.zeros(len(self.states), len(self.actions))
        Q += self.reward_function.reward
        for x in self.states:
            for a in self.actions:
                Q[x, a] += self.gamma * np.array([self.dynamics[x, a] for x in self.states]).dot(values)
        return Q

    def find_optimal_policy(self):
        """Must compute the optimal policy thanks to the true reward function"""
        optimal_policy = np.zeros(self.nb_states)

        ### ***** For Value iteration *****
        '''
            Implement the bellman_operator_optimal : W --> T*(W), given Reward, Proba_transition, and discount_factor
        '''

        def bellman_operator_optimal(W, actions, R, P, gamma):
            reward = np.array([R[x] for x in range(len(W))])
            t = np.array([[np.dot(P[x, a], W) for a in actions] for x in range(len(W))])
            u = reward + gamma * t

            return np.max(u, axis=1)

        '''
            Implement the value iteration method thanks to the bellman operator.

            Input : - R reward for (state,action)
                    - actions the actions space
                    - P proba of transition
                    - gamma discount_factor
                    - nb_step : the number of iterations we do.
            Output: - The optimal policy pi
        '''

        def value_iteration(R, actions, P, gamma, nb_step):
            nb_state_possible = len(P)
            #  W_over_time = np.zeros((nb_step,nb_state_possible))
            W = np.zeros(nb_state_possible)

            for k in range(nb_step):
                W = bellman_operator_optimal(W, actions, R, P, gamma)
                #  W_over_time[k] = W

            t = np.array([[np.dot(P[x, a], W) for a in actions] for x in range(nb_state_possible)])
            u = R + gamma * t

            pi = Policy(np.array([np.argmax(u[i]) for i in range(nb_state_possible)]))
            #  V = np.array([u[x, pi[x]] for x in range(nb_state_possible)])
            return pi

        optimal_policy = value_iteration(self.true_reward, self.actions, self.dynamics, self.gamma, nb_step=100)
        return optimal_policy
