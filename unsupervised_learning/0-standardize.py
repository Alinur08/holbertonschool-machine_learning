#!/usr/bin/env python3
"""
Defines a function that standardizes a dataset using Scikit-learn
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes a tabular dataset (n_samples, n_features)

    Parameters:
        X (numpy.ndarray): The input data to be standardized

    Returns:
        numpy.ndarray: The standardized dataset
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
