#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    l_list = []
    b_list = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or \
                log_l is None:
            return None, None, None, None

        results.append((pi, m, S))
        l_list.append(log_l)

        p = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * log_l
        b_list.append(bic)

    l_arr = np.array(l_list)
    b_arr = np.array(b_list)

    best_index = np.argmin(b_arr)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, l_arr, b_arr
