#!/usr/bin/env python3
"""Calculate the expectation step in the EM algorithm for a GMM."""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X (np.ndarray): of shape (n, d)
            containing the dataset
        pi (np.ndarray): of shape (k,)
            containing the priors for each cluster
        m (np.ndarray): of shape (k, d)
            containing the centroïde means for each cluster
        S (np.ndarray): of shape (k, d, d)
            containing the covariance matrices for each cluster

    Return:
        g, l or None, None on failure
            g (np.ndarray): of shape (k, n)
                containing the posterior probabilities
                for each data point in each cluster
            l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    # Pour chaque cluster
    for cluster in range(k):
        # calcule à quel point chaque point est proche
        densities = pdf(X, m[cluster], S[cluster])
        # Multiplié par pi
        g[cluster] = pi[cluster] * densities

    # Calcule des pourcentages
    total = np.sum(g, axis=0)
    g = g / total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
