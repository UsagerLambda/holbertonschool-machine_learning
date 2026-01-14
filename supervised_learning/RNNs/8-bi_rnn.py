#!/usr/bin/env python3
"""Bidirectional RNN."""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Bidirectional RNN."""
    t, m, i = X.shape
    H_forward = [h_0]
    H_backward = [h_t]

    for step in range(t):
        h_next = bi_cell.forward(H_forward[step], X[step])
        H_forward.append(h_next)

    for step in range(t - 1, -1, -1):
        h_prev = bi_cell.backward(H_backward[-1], X[step])
        H_backward.append(h_prev)

    # Supprime h_0
    H_forward = np.array(H_forward[1:])
    # Supprime h_t
    H_backward = np.array(list(reversed(H_backward))[1:])
    H = np.concatenate((H_forward, H_backward), axis=2)
    Y = bi_cell.output(H)
    return H, Y
