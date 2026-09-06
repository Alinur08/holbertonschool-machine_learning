#!/usr/bin/env python3
"""Module that computes the Monte-Carlo policy gradient."""
import numpy as np


def policy(matrix, weight):
    """
    Compute the policy with a weight of a matrix.
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient of a state and weight matrix.
    """
    state = state.reshape(1, -1) if state.ndim == 1 else state

    P = policy(state, weight)
    action = np.random.choice(P.shape[1], p=P[0])

    s = P.reshape(-1, 1)
    softmax_derivative = np.diagflat(s) - s.dot(s.T)
    dsoftmax = softmax_derivative[action, :]
    dlog = dsoftmax / P[0, action]
    gradient = state.T.dot(dlog[np.newaxis, :])

    return action, gradient
