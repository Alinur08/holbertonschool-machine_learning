#!/usr/bin/env python3
"""
Defines a function that creates and fits a K-Means model using Scikit-learn
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Creates and fits a K-Means clustering model on tabular data

    Parameters:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_clusters (int): Number of clusters to find
        random_state (int): Random seed for reproducibility

    Returns:
        sklearn.cluster.KMeans: Fitted K-Means instance
    """
    kmeans = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    kmeans.fit(X)
    return kmeans
