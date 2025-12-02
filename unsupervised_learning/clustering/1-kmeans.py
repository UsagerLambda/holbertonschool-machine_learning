#!/usr/bin/env python3
"""Performs K-means function."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int, number of clusters
        iterations: int, maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means
        clss: numpy.ndarray of shape (n,) containing the index of the cluster
        Returns (None, None) if the function fails
    """
    if not isinstance(k, int) or k <= 0:
        return None, None
    try:
        # Crée des centroïdes dans les limites min, max des données
        C = np.random.uniform(
                np.min(X, axis=0),
                np.max(X, axis=0),
                (k, X.shape[1])
            )

        for i in range(iterations):
            Cprev = C.copy()  # Sauvegarde des centroïdes actuels

            # Calcule la distance euclidienne entre chaque point X et
            # chaque centroïde C
            distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

            # Attribut les points dans le cluster du centroïde le plus proche
            clss = distances.argmin(axis=1)

            # Mise à jour des centroïdes
            for j in range(k):
                # Récupère les points du cluster j dans la range clss
                pts = X[clss == j]
                if len(pts) > 0:
                    # Fait la moyenne et met a jour le centroïde
                    C[j] = pts.mean(axis=0)
                else:
                    # Sinon replacer le centroïde de manière aléatoire
                    C[j] = np.random.uniform(
                        np.min(X, axis=0),
                        np.max(X, axis=0),
                        X.shape[1]
                    )

            # Si les centroïdes n'ont pas ou
            # presque pas bougé à la dernière itération
            if np.allclose(C, Cprev):
                break  # Sortir de la boucle

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = distances.argmin(axis=1)

        return C, clss

    except Exception:
        return None, None
