#!/usr/bin/env python3
"""Convert one-hot matrix into numeric label vector."""

import numpy as np


def one_hot_decode(one_hot):
    """Convert one-hot matrix into numeric label vector.

    Args:
        one_hot (numpy.ndarray): one-hot array representation

    Returns:
        list: Decoded One-Hot
    """
    return np.argmax(one_hot, axis=0)
