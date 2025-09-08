#!/usr/bin/env python3
"""Calcul la moyenne et l'écart-type des features."""

import numpy as np


def normalization_constants(X):
    """
    Calculate la moyenne et l'écart-type de chaque feature (colonne) de X.

    Args:
        X: np.ndarray de forme (m, n)
    Returns:
        mean: np.ndarray de forme (n,)
        std: np.ndarray de forme (n,)
    """
    # Calcul la moyenne par colonnes (une moyenne / features)
    mean = np.mean(X, axis=0)
    # Calcul l'écart-type par colonne (écart-type / features)
    std = np.std(X, axis=0)
    return mean, std
