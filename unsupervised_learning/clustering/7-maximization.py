#!/usr/bin/env python3
"""Calculate Maximization step in GMM"""
import numpy as np


def maximization(X, g):
    """
    Calculates the Maximization step in GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    N_k = np.sum(g, axis=1)

    pi = N_k / n

    m = (g @ X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = (g[i][:, np.newaxis] * diff).T @ diff / N_k[i]

    return pi, m, S
