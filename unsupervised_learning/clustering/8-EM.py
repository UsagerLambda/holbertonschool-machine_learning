#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the expectation maximization for a GMM.

    Args:
        X (np.ndarray): of shape (n, d) containing the data set
        k (int): positive integer containing the number of clusters
        iterations (int, optional): positive integer
            containing the maximum number of iterations for the algorithm
        tol (float, optional): is a non-negative float
            containing tolerance of the log likelihood,
            used to determine early stopping i.e.
            if the difference is less than or equal to tol,
            it should stop the algorithm
        verbose (bool, optional): is a boolean
            that determines if you should print information about the algorithm

    Returns:
        pi, m, S, g, l: or None, None, None, None, None on failure
            pi is a numpy.ndarray of shape (k,) containing the priors
            m is a numpy.ndarray of shape (k, d) containing the centroid means
            S is a numpy.ndarray of shape (k, d, d) containing covariances
            g is a numpy.ndarray of shape (k, n) containing the probabilities
            l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialize les parametres
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    # Expectation Step
    g, log_likelihood = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    prev_log_likelihood = log_likelihood

    # EM algo
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after \
                {i} iterations: {round(log_likelihood, 5)}")

        # Maximization step
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        # Expectation Step
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if abs(log_likelihood - prev_log_likelihood) <= tol:
            if verbose:
                print(f"Log Likelihood after \
                    {i + 1} iterations: {round(log_likelihood, 5)}")
            break

        prev_log_likelihood = log_likelihood
    else:
        if verbose:
            print("Log Likelihood after {} iterations: {}".format(
                iterations, round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
