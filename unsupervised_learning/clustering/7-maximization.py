#!/usr/bin/env python3
"""Calculate the maximization step in the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X (np.ndarray): of shape (n, d) containing the data set
        g (np.ndarray): of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    # Proportion des points dans chaque cluster
    pi = np.sum(g, axis=1) / n

    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    # Calcul le centre de chaque cluster et la forme des clusters
    for cluster in range(k):
        m[cluster] = np.sum(g[
            cluster, :, np.newaxis] * X, axis=0) / np.sum(g[cluster])

        diff = X - m[cluster]
        weighted = g[
            cluster, :, np.newaxis, np.newaxis] * (
                diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])
        S[cluster] = np.sum(weighted, axis=0) / np.sum(g[cluster])

    return pi, m, S
