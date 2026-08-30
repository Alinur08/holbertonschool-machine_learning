#!/usr/bin/env python3
"""Kmeans Clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    low = X.min(axis=0)
    high = X.max(axis=0)
    centroids = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.copy(centroids)
        for i in range(k):
            points = X[labels == i]
            if points.shape[0] == 0:
                new_centroids[i] = np.random.uniform(low, high, size=(d,))
            else:
                new_centroids[i] = points.mean(axis=0)

        if np.all(centroids == new_centroids):
            return centroids, labels

        centroids = new_centroids

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    return centroids, labels
