#!/usr/bin/env python3
"""
Defines a function that performs PCA on tabular data using Scikit-learn
"""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
    Performs Principal Component Analysis (PCA) on tabular data

    Parameters:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_components (int | float | None): Number of components or fraction
                                            of variance to preserve
        random_state (int): Random seed for reproducibility

    Returns:
        numpy.ndarray: Data transformed into principal component space
        sklearn.decomposition.PCA: Fitted PCA instance
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)
    return X_pca, pca
