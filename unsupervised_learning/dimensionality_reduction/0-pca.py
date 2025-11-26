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
    _, S, Vt = np.linalg.svd(X)

    variance = (S ** 2) / np.sum(S ** 2)
    cumulative_var = np.cumsum(variance)

    nd = np.where(cumulative_var >= var)[0][0] + 2

    return Vt[:nd].T
