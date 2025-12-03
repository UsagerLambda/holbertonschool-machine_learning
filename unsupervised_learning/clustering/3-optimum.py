#!/usr/bin/env python3
"""Function that tests for the optimum number of clusters by variance."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test for the optimum number of clusters by variance.

    Args:
        X (np.ndarray): _description_
        kmin (int, optional): Positive integer containing
            the minimun number of clusters. Defaults to 1.
        kmax (int, optional): Positive integer containing
            the maximum number of clusters. Defaults to None.
        iterations (int, optional): Maximum number of iterations for K-means.
            Defaults to 1000.

    Returns:
        results, d_vars, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or X.shape[0] < 1:
        return None, None

    if not isinstance(kmin, int) or not isinstance(iterations, int):
        return None, None

    if kmax is not None and not isinstance(kmax, int):
        return None, None

    if kmax is None:
        kmax = len(X)

    if kmin <= 0 or kmax < kmin or iterations <= 0:
        return None, None

    if kmax > X.shape[0] or kmax - kmin < 1:
        return None, None

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, clss))
        d_vars.append(var)

    variance_ref = d_vars[0]

    for i in range(len(d_vars)):
        d_vars[i] = variance_ref - d_vars[i]

    return results, d_vars
