#!/usr/bin/env python3
"""Performs a t-SNE transformation"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation

    X is a numpy.ndarray of shape (n, d) containing the dataset to be
        transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    ndims is the new dimensional representation of X
    idims is the intermediate dimensional representation of X after PCA
    perplexity is the perplexity
    iterations is the number of iterations
    lr is the learning rate

    Returns: Y, a numpy.ndarray of shape (n, ndim) containing the
        optimized low dimensional transformation of X
    """
    n, d = X.shape
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    P = P * 4
    Y = np.random.randn(n, ndims)
    Y_prev = Y.copy()

    for i in range(iterations):
        dY, Q = grads(Y, P)

        if i < 20:
            a = 0.5
        else:
            a = 0.8

        if i == 0:
            Y_new = Y - lr * dY
        else:
            Y_new = Y - lr * dY + a * (Y - Y_prev)

        Y_prev = Y
        Y = Y_new
        Y = Y - np.mean(Y, axis=0)

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        if i == 99:
            P = P / 4

    return Y
