#!/usr/bin/env python3
"""Calculates the intra-cluster variance of a dataset"""
import numpy as np


def variance(X, C):
    """
    Calculates the intra-cluster variance of a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    labels = np.argmin(distances, axis=1)

    if C.shape[0] == 0:
        return 0.0

    assigned_centroids = C[labels]
    return np.sum(np.sum((X - assigned_centroids) ** 2, axis=1))
