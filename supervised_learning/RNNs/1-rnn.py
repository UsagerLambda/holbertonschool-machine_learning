#!/usr/bin/env python3
"""A function that performs forward propagation for a simple RNN."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN."""
    t, m, i = X.shape  # 6, 8, 10
    H = [h_0]
    Y = []
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H.append(h_next)
        Y.append(y)
    return np.array(H), np.array(Y)
