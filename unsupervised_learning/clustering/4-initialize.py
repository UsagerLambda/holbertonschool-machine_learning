#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize variables for a Gaussian Mixture Model.

    Args:
        X (np.dnarray):  of shape (n, d) containing the data set
        k (int): positive integer containing the number of clusters

    Return:
        pi is a np.ndarray of shape (k,)
            containing the priors for each cluster,
            initialized evenly
        m is a np.ndarray of shape (k, d)
            containing the centroid means for each cluster,
            initialized with K-means
        S is a np.ndarray of shape (k, d, d)
            containing the covariance matrices for each cluster,
            initialized as identity matrices
    """
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    _, d = X.shape

    pi = np.full(k, 1/k)

    m, _ = kmeans(X, k, 1000)

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
