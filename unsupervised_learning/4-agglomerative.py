#!/usr/bin/env python3
"""
Defines a function that performs Agglomerative hierarchical clustering
"""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(
    X, n_clusters, random_state, n_components, use_pca_data=True
):
    """
    Performs Agglomerative hierarchical clustering on tabular data

    Parameters:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        n_components (int): Number of PCA components to reduce data to
        use_pca_data (bool): Whether to apply PCA before clustering

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted model instance
        numpy.ndarray: Data used for fitting
        float: Silhouette score of the clustering
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components, random_state)
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    labels = model.fit_predict(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, labels)
    else:
        score = None

    return model, X_used, score
