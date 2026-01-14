#!/usr/bin/env python3
"""Class that represents a Bidirectional RNN."""

import numpy as np


class BidirectionalCell:
    """Class that represents cells of a Bidirectional RNN."""

    def __init__(self, i, h, o):
        """Class constructor."""
        self.Whf = np.random.randn(h + i, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(h + i, h)
        self.bhb = np.zeros((1, h))

        # fois 2 car foward + backward
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward propagation."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """Backward cell."""
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """Calculates all outputs for the RNN."""
        val = np.exp(H @ self.Wy + self.by)
        Y = val / np.sum(val, axis=2, keepdims=True)
        return Y