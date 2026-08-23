#!/usr/bin/env python3
"""Performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X

    Returns: T, a numpy.ndarray of shape (n, ndim) containing the
        transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T
    T = np.matmul(X_m, W)
    return T
