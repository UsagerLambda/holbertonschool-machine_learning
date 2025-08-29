#!/usr/bin/env python3
"""Convert numeric label vector into one-hot matrix."""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Encode un vecteur d'étiquettes entières en représentation one-hot.

    Args:
        Y (np.ndarray): Tableau d'entiers de forme (m,) ou (m, 1),
            contenant les étiquettes à encoder.
        classes (int): Nombre total de classes possibles.

    Returns:
        np.ndarray ou None: Matrice one-hot de forme (m, classes),
            ou None si une erreur survient.
    """
    try:
        # Crée une matrice one-hot :
        # pour chaque valeur de Y,
        # place un 1 à l’index correspondant dans une ligne de taille classes,
        # le reste à 0.
        one_hot = np.eye(classes)[Y].T
    except Exception:
        return None
    return one_hot
