#!/usr/bin/env python3
"""Calculate the mean and covariance of a data set."""

import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance of a data set."""
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / len(X)
    return mean, cov
