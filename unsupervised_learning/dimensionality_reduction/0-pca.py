#!/usr/bin/env python3
"""Performs PCA on a dataset."""

import numpy as np


def pca(X, var=0.95):
    """
    Create function that performs PCA on a dataset.

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
    cov_matrix = np.matmul(X.T, X) / (X.shape[0] - 1)
    
    # Calculer les valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Trier par valeurs propres décroissantes
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculer la variance expliquée cumulative
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    
    # Trouver le nombre de composantes nécessaires
    nd = np.argmax(cumulative_variance >= var) + 1
    
    # Retourner les nd premiers vecteurs propres
    return eigenvectors[:, :nd]
