#!/usr/bin/env python3
"""Calculate la variance intra-cluster."""

import numpy as np


def variance(X, C):
    """
    Calculate la variance intra-cluster.

    Args:
        X (np.ndarray): de shape (n, d) -> n points, d dimensions
        C (np.ndarray): de shape (k, d) -> k centroïdes, d dimensions

    Returns:
        float: variance intra-cluster totale, ou None si erreur
    """
    # Variance intra-cluster = somme des distances au carré entre chaque point
    # et son centroïde le plus proche

    # np.linalg.norm(X[:, np.newaxis] - C, axis=2) calcule les distances
    # entre tous les points et tous les centroïdes

    # np.min(..., axis=1) trouve la distance minimale
    # (centroïde le plus proche) pour chaque point

    # ** 2 élève au carré ces distances minimales
    # np.sum() fait la somme totale de toutes ces distances au carré
    try:
        return np.sum(
            np.min(np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1) ** 2)
    except Exception:
        return None
