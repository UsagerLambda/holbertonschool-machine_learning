#!/usr/bin/env python3
"""A function that performs a forward propagation for a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN."""
    t, m, i = X.shape
    long = len(rnn_cells)
    h_prev = h_0.copy()
    H = [h_0]
    Y = []
    for step in range(t):
        x_input = X[step]
        h_new = []
        for layer in range(long):
            h_next, y = rnn_cells[layer].forward(h_prev[layer], x_input)
            h_new.append(h_next)
            x_input = h_next
        h_prev = np.array(h_new)
        H.append(h_prev)
        Y.append(y)

    return np.array(H), np.array(Y)
