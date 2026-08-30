#!/usr/bin/env python3
"""Full EM algorithm for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, prev_l = expectation(X, pi, m, S)
    if g is None or prev_l is None:
        return None, None, None, None, None

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(prev_l, 5)))

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        if abs(log_likelihood - prev_l) <= tol:
            prev_l = log_likelihood
            break

        prev_l = log_likelihood

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i + 1, round(prev_l, 5)))

    return pi, m, S, g, prev_l
