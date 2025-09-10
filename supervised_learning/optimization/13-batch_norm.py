#!/usr/bin/env python3
"""Normalizes an unactivated output of a neural network."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output of a neural network.

    Args:
        Z (numpy.ndarray): matrix of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in z
        gamma (numpy.ndarray): matrix of shape (1, n),
            containing the scales used for batch normalization
        beta (numpy.ndarray): matrix of shape (1, n),
            containing the offsets used for batch normalization
        epsilon (float): small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    # Calculer la moyenne pour chaque feature
    mu = np.mean(Z, axis=0)
    # Calculer la variance pour chaque feature
    variance = np.var(Z, axis=0)
    # centre chaque feature autour de 0 puis divise par l'écart-type
    Z_norm = (Z - mu) / np.sqrt(variance + epsilon)
    # Réajustement avec les paramètres
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
