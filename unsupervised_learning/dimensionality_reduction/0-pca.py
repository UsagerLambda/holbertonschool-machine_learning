#!/usr/bin/env python3
"""text"""

import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset.

    Args:
        X (np.ndarray): Array of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each points
            all dimensions have a mean of 0 across all data points.
        var (float, optional): is the fraction of the variance
            that the PCA transformation should maintain. Defaults to 0.95.

    Returns:
        W (np.ndarray): Array of shape (d, nd)
            where nd is the new dimensionality of the transformed X
    """
    # Singular Value Decomposition (SVD)
    # SVD = outil mathématique qui décompose X en 3 matrices
    # U = directions dans l'espace des échantillons (non utilisé ici)
    # S = valeurs singulières (force de chaque direction)
    # Vt = directions principales (axes de variance maximale)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Calcul de la variance expliquée par chaque direction
    # S² = variance brute, divisée par la somme = pourcentage de variance
    # Ex: [0.7, 0.2, 0.05, 0.03, 0.02] = première direction capture 70%
    explained_variance = (S ** 2) / np.sum(S ** 2)

    # Variance cumulée : somme progressive des variances
    # Ex: [0.7, 0.9, 0.95, 0.98, 1.0]
    # Permet de voir combien de directions sont nécessaires pour atteindre var%
    cumulative_variance = np.cumsum(explained_variance)

    # Trouve le nombre minimum de dimensions (nd) pour capturer var% de variance
    # argmax trouve le premier indice où cumulative_variance >= var
    # +1 car les indices Python commencent à 0
    nd = np.argmax(cumulative_variance >= var) + 1

    # Retourne les nd premières directions principales (colonnes de Vt transposé)
    # Vt[:nd] = garde les nd premières lignes (directions les plus importantes)
    # .T = transpose pour avoir le format (d, nd) attendu
    return Vt[:nd].T
