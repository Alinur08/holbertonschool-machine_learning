#!/usr/bin/env python3
"""Defines the rnn function that performs forward propagation"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    """
    t = X.shape[0]
    m, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y = []

    h_prev = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_next
        Y.append(y)
        h_prev = h_next

    Y = np.array(Y)

    return H, Y
