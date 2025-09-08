#!/usr/bin/env python3
"""Permute the X and Y matrix."""

import numpy as np


def shuffle_data(X, Y):
    """
    Mélange X et Y de façon synchrone selon la première dimension.

    Args:
        X (np.ndarray): Données d'entrée de forme (m, ...).
        Y (np.ndarray): Labels ou cibles de forme (m, ...).

    Returns:
        tuple: X et Y mélangés dans le même ordre.
    """
    perm = np.random.permutation(X.shape[0])  # Permute par rapport à X
    # Applique la permutation de X pour X et Y
    # (le même ordre de permutation sera appliqué pour Y)
    return X[perm], Y[perm]
