#!/usr/bin/env python3
"""Calculate PDF of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the PDF of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if (
        X.shape[1] != m.shape[0]
        or m.shape[0] != S.shape[0]
        or S.shape[0] != S.shape[1]
    ):
        return None

    n, d = X.shape

    det_S = np.linalg.det(S)
    if det_S == 0:
        return None

    inv_S = np.linalg.inv(S)

    diff = X - m
    exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)

    pdf_values = (1 / np.sqrt((2 * np.pi) ** d * det_S)) * np.exp(exponent)

    return np.maximum(pdf_values, 1e-300)
