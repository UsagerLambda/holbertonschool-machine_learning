#!/usr/bin/env python3
"""Calculate the mean and covariance of a data set."""

import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance of a data set."""
    if len(X) != 2:
        raise TypeError("data must be a 2D numpy.ndarray")
    if len(X.shape[0]) < 2:
        raise ValueError("data must contain multiple data points")
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / len(X)
    return mean, cov
