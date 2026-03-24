#!/usr/bin/env python3
"""Module de politique par gradient pour l'apprentissage par renforcement."""

import numpy as np


def policy(matrix, weight):
    """Calculate la politique stochastique via softmax sur les scores pondérés.

    Args:
        matrix: numpy.ndarray de forme (n,) représentant l'état courant.
        weight: numpy.ndarray de forme (n, m) représentant la matrice de
            poids, où m est le nombre d'actions possibles.

    Returns:
        numpy.ndarray de forme (m,) contenant la distribution de probabilité
        sur les actions.
    """
    # state * weight donne un score par action
    z = np.dot(matrix, weight)
    # Rend positif + emplifie les écarts
    exp_z = np.exp(z)
    # Divise par le total -> somme à 1 -> proba
    probs = exp_z / np.sum(exp_z)
    return probs
