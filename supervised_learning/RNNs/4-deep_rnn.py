#!/usr/bin/env python3
"""Defines the deep_rnn function that performs forward propagation
for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    """
    t = X.shape[0]
    l, m, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = []

    for step in range(t):
        x = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
