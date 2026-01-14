#!/usr/bin/env python3
"""Class that represents a cell of a Bidirectional RNN."""

import numpy as np


class BidirectionalCell:
    """Class that represents a cell of a Bidirectional RNN."""

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
        """Forward propagation using softmax."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next
