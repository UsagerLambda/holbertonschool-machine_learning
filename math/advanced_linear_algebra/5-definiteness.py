#!/usr/bin/env python3
"""Calculates the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Calculate the definiteness of a matrix."""
    # Vérifie que la matrice est bien de type np.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Vérifie si la matrice est carré
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        return None

    # Vérifie si la matrice est 2D
    if len(matrix) != 2:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigvals = np.linalg.eigvals(matrix)

    if np.all(eigvals > 0):
        return "Positive definite"
    elif np.all(eigvals >= 0):
        return "Positive semi-definite"
    elif np.all(eigvals < 0):
        return "Negative definite"
    elif np.all(eigvals <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
