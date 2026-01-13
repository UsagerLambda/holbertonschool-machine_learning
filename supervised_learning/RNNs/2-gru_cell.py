#!/usr/bin/env python3
"""Class GRUCell that represents a cell of a GRU RNN."""

import numpy as np


class GRUCell:
    """Class GRUCell that represent a cell of a GRU RNN."""

    def __init__(self, i, h, o):
        """Class constructor."""
        self.Wz = np.random.randn(h + i, h)
        self.bz = np.zeros((1, h))

        self.Wr = np.random.randn(h + i, h)
        self.br = np.zeros((1, h))

        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward propagation of a GRU RNN."""
        def sigmoid(x):
            """Sigmoid function."""
            return 1 / (1 + np.exp(-x))

        def softmax(h, w, b):
            """Softmax function."""
            val = np.exp(h @ w + b)
            y = val / np.sum(val, axis=1, keepdims=True)
            return y

        concat = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(concat @ self.Wz + self.bz)
        r = sigmoid(concat @ self.Wr + self.br)

        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(concat_reset @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde
        y = softmax(h_next, self.Wy, self.by)

        return h_next, y
