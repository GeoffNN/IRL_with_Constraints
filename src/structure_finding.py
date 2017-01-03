"""Refining reward functions, using a priori structure on reward"""

import numpy as np
from cvxopt import matrix, solvers


def objective_matrix_nonneg(mdp, reward):
    A, _ = constraints_nonneg(mdp, reward)
    return A.T * A


def constraints_nonneg(mdp, reward):
    """Computes the constraint inequalities in canonic form for the qp solver"""
    # TODO: extract A and b methods
    # Supposes that states are indexed from 0 to nb_states-1
    G = np.zeros((mdp.nb_states ** 2, mdp.nb_states))
    for k, s in enumerate(mdp.states):
        G[k * mdp.nb_states:(k + 1) * mdp.nb_states, s] += np.ones(mdp.nb_states)
        G[k * mdp.nb_states:(k + 1) * mdp.nb_states, :] -= mdp.gamma * np.eye(mdp.nb_states)
    h = reward.flatten()
    return matrix(G), matrix(h)


def compute_potential_nonneg(mdp, reward):
    P = objective_matrix_nonneg(mdp, reward)
    q = matrix(np.zeros(mdp.nb_states))
    G, h = constraints_nonneg(mdp, reward)
    sol = solvers.qp(P, q, G, h)
    return sol['x']


def compute_reward_nonneg(mdp, reward):
    phi = compute_potential_nonneg(mdp, reward)
    return np.array([[mdp.gamma * r1 - r2 for r1 in phi] for r2 in phi])
