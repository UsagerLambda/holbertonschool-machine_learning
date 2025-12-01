#!/usr/bin/env python3
"""Initialize K-means function."""

import numpy as np


def initialize(X, k):
    """
    Initialize clusters centroids for K-means.

    Args:
        X (np.ndarray): Array of shape (n, d) containing the dataset that
        will be used for K-means clustering.
            - n is the number of data points
            - d is the number of dimensions for each data point
        k (integer): Positive integer containing the number of clusters
    """
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        return np.random.uniform(
                np.min(X, axis=0),  # valeur min de X pour chaque dimension
                np.max(X, axis=0),  # valeur max de X pour chaque dimension
                (k, X.shape[1])  # Forme de retour, k centroïdes de dimension d
            )
    except Exception:
        return None
