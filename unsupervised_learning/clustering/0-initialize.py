#!/usr/bin/env python3
"""Kmeans Clustering initialization of centroids"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, d))
