#!/usr/bin/env python3
"""Calculate a correlation matrix."""

import numpy as np


def correlation(C):
    """Calculate a correlation matrix."""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # ρ(i,j) = Cov(i,j) / (σ_i × σ_j)

    # Calcul la racine carré de chaque éléments dans la diagonal de la matrice
    std_devs = np.sqrt(np.diag(C))

    # Calcul le produit extérieur
    outer_std = np.outer(std_devs, std_devs)

    # Calcul la correlation
    correlation_matrix = C / outer_std

    return correlation_matrix
