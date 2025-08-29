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
    if not isinstance(one_hot, np.ndarray):
        return None
    S = np.argmax(one_hot, axis=0)

    if np.all(S == 0):
        return None

    return S
