#!/usr/bin/env python3
"""Multivariate Normal distribution."""

import numpy as np


class MultiNormal:
    """Multivariate Normal distribution."""

    def __init__(self, data):
        """Class constructor of a Multivariate Normal distribution."""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculate the PDF at a data point."""
        # Vérifier que x est un numpy.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Obtenir le nombre de dimensions d depuis self.mean
        d = self.mean.shape[0]

        # Vérifier que x a la forme correcte (d, 1)
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculer l'inverse et le déterminant de la matrice de covariance
        cov_inv = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)

        # Calculer la constante de normalisation : 1 / sqrt((2π)^d * |Σ|)
        norm_constant = 1.0 / np.sqrt((2 * np.pi) ** len(self.mean) * det_cov)

        # Calculer la différence entre x et la moyenne
        diff = x - self.mean

        # Calculer l'exposant : -0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ)
        exponent = -0.5 * diff.T @ cov_inv @ diff

        # Retourner la valeur du PDF en tant que scalaire
        return norm_constant * float(np.exp(exponent))
