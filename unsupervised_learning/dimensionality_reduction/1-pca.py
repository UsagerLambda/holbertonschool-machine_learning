#!/usr/bin/env python3
"""Performs PCA on a dataset."""

import numpy as np


def pca(X, ndim):
    """
    Create function that performs PCA on a dataset.

    Args:
        X (np.ndarray): Array of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each points
            all dimensions have a mean of 0 across all data points.
        ndim (int): the new dimensionality of the transformed X

    Returns:
        W (np.ndarray): Array of shape (d, ndim)
            containing the projection matrix
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    W = Vt[:ndim].T

    T = np.matmul(X_centered, W)

    return T
