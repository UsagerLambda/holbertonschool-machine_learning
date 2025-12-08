#!/usr/bin/env python3
"""Init a Gaussian Process class."""

import numpy as np


class GaussianProcess:
    """Représente un processus gaussien pour l'optimisation bayésienne."""

    def __init__(self, X_init, Y_init, length=1, sigma_f=1):
        """
        Initialise un processus gaussien.

        Args:
            X_init (np.ndarray): de forme (t, 1) représentant
                les entrées déjà échantillonnées avec la fonction boîte noire
                - t est le nombre d'échantillons initiaux
            Y_init (np.ndarray): de forme (t, 1) représentant
                les sorties de la fonction boîte noire pour chaque
                entrée dans X_init
                - t est le nombre d'échantillons initiaux
            length (int, optional): paramètre de longueur pour le noyau
            sigma_f (int, optional): l'écart type donné
                à la sortie de la fonction boîte noire
        """
        self.X = X_init
        self.Y = Y_init
        self.length = length
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices.

        Args:
            X1 (np.ndarray): de forme (m, 1)
            X2 (np.ndarray): de forme (m, 1)

        Returns:
            la matrice de noyau de covariance comme un numpy.ndarray
            de forme (m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(
            X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.length**2 * sqdist)
