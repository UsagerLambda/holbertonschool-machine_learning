#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the BIC."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using the BIC.

    Args:
        X (numpy.ndarray): Array of shape (n, d) containing the data set.
        kmin (int): Positive integer containing the minimum number of clusters
            to check for (inclusive). Default is 1.
        kmax (int): Positive integer containing the maximum number of clusters
            to check for (inclusive). If None, kmax is set to the maximum
            number of clusters possible. Default is None.
        iterations (int): Positive integer containing the maximum number of
            iterations for the EM algorithm. Default is 1000.
        tol (float): Non-negative float containing the tolerance for the EM
            algorithm. Default is 1e-5.
        verbose (bool): Boolean that determines if the EM algorithm should
            print information to the standard output. Default is False.

    Returns:
        tuple: (best_k, best_result, l, b) or (None, None, None, None) on
            failure, where:
            - best_k (int): The best value for k based on its BIC.
            - best_result (tuple): Tuple containing (pi, m, S) where:
                - pi (numpy.ndarray): Array of shape (k,) containing the
                  cluster priors for the best number of clusters.
                - m (numpy.ndarray): Array of shape (k, d) containing the
                  centroid means for the best number of clusters.
                - S (numpy.ndarray): Array of shape (k, d, d) containing the
                  covariance matrices for the best number of clusters.
            - l (numpy.ndarray): Array of shape (kmax - kmin + 1) containing
              the log likelihood for each cluster size tested.
            - b (numpy.ndarray): Array of shape (kmax - kmin + 1) containing
              the BIC value for each cluster size tested.
              BIC = p * ln(n) - 2 * l, where:
                - p is the number of parameters required for the model
                - n is the number of data points used to create the model
                - l is the log likelihood of the model
    """
    # BIC = log-likelihood - (p/2) × log(n)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    if not isinstance(kmin, int) or not isinstance(iterations, int):
        return None, None, None, None

    if kmax is not None and not isinstance(kmax, int):
        return None, None, None, None

    if kmax is None:
        kmax = len(X)

    if kmin <= 0 or kmax < kmin or iterations <= 0:
        return None, None, None, None

    if kmax > X.shape[0] or kmax - kmin < 1:
        return None, None, None, None

    d = X.shape[1]
    n = X.shape[0]

    results = []
    b = []
    likes = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        p = (k - 1) + k * d + k * (d * (d+1) // 2)
        b.append(p * np.log(n) - 2 * log_likelihood)
        likes.append(log_likelihood)
        results.append((pi, m, S))

    best_k = kmin + np.argmin(b)  # argmin renvois un index
    best_result = results[np.argmin(b)]  # Donne résultats du meilleur index
    likes = np.array(likes)
    b = np.array(b)
    return best_k, best_result, likes, b
