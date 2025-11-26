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
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_variance = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    indices = np.where(cumulative_variance >= var)[0]
    if len(indices) > 0:
        nd = indices[0] + 1
    else:
        nd = len(cumulative_variance)

    return Vt[:nd].T
