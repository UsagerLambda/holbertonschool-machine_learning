#!/usr/bin/env python3
"""Init a Gaussian Process class."""

import numpy as np


class GaussianProcess:
    """Représente un processus gaussien pour l'optimisation bayésienne."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
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
            l (int, optional): paramètre de longueur pour le noyau
            sigma_f (int, optional): l'écart type donné
                à la sortie de la fonction boîte noire
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
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
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predit la moyenne et l'écart type des points d'un processus gaussien.

        Args:
            X_s (np.ndarray): de forme (s, 1) contenant tous les points
                dont la moyenne et l'écart type doivent être calculés
                - s : est le nombre points d'échantillonage

        Returns:
            mu, sigma
            - mu (np.ndarray): de forme (s, ) contenant les moyennes
                pour chaque points dans X_s respectivement.
            - sigma (np.ndarray): de forme (s, ) contenant les variances
                pour chaque points dans X_s respectivement.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)
        mu = K_s.T @ K_inv @ self.Y
        sigma = K_ss - K_s.T @ K_inv @ K_s

        return mu.flatten(), np.diag(sigma)

    def update(self, X_new, Y_new):
        """
        Met à jour le processus gaussien avec un nouvel échantillon.

        Args:
            X_new (np.ndarray): de forme (1,) représentant
                le nouveau point d'échantillonnage
            Y_new (np.ndarray): de forme (1,) représentant
                la nouvelle valeur d'échantillonnage
        """
        self.X = np.vstack([self.X, X_new])
        self.Y = np.vstack([self.Y, Y_new])
        self.K = self.kernel(self.X, self.X)
