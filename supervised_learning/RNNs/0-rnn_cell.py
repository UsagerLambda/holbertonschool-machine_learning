#!/usr/bin/env python3
"""Class RNNCell that represents a cell of a simple RNN."""

import numpy as np


class RNNCell:
    """Class RNNCell that represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Class constructor."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward propagation using softmax."""
        # Concatenate pour donner au neurone la vision
        # de ce qu'il a vu avant et maintenant.
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calcule le nouvel état caché (la nouvelle mémoire)
        # tanh pour garder les valeurs entre -1 & 1
        h_next = np.tanh(concat @ self.Wh + self.bh)
        # Transforme l'état caché en scores pour chaque classe
        # puis softmax pour convertir en probabilités
        val = np.exp(h_next @ self.Wy + self.by)
        y = val / np.sum(val, axis=1, keepdims=True)
        # Renvoie la nouvelle mémoire & les probabilités
        return h_next, y
