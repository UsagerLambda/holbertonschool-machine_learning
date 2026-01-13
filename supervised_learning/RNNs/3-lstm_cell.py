#!/usr/bin/env python3
"""Class GRUCell that represents a cell of a LSTM RNN."""

import numpy as np


class LSTMCell:
    """Class GRUCell that represents a cell of a LSTM RNN."""

    def __init__(self, i, h, o):
        """Class constructor."""
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))

        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))

        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))

        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Forward propagation of a LSTM RNN."""

        def sigmoid(x):
            """Sigmoid function."""
            return 1 / (1 + np.exp(-x))

        def softmax(h, w, b):
            """Softmax function."""
            val = np.exp(h @ w + b)
            y = val / np.sum(val, axis=1, keepdims=True)
            return y

        concat = np.concatenate((h_prev, x_t), axis=1)

        input_gate = sigmoid(concat @ self.Wu + self.bu) * np.tanh(
            concat @ self.Wc + self.bc
        )
        c_next = (c_prev * sigmoid(concat @ self.Wf + self.bf)) + input_gate
        h_next = sigmoid(concat @ self.Wo + self.bo) * np.tanh(c_next)

        y = softmax(h_next, self.Wy, self.by)

        return h_next, c_next, y
