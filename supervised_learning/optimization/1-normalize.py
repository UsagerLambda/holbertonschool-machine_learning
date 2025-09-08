#!/usr/bin/env python3
"""Normalized the X matrix."""

import numpy as np


def normalize(X, m, s):
    """Normalize a matrix.

    Args:
        X (numpy.ndarray): matrix of shape (d, nx) to normalize
            - d is the number of data points
            - nx is the number of features
        m (numpy.ndarray): matrix of shape (nx,)
            that contains the mean of all features of X
        s (numpy.ndarray): matrix of shape (nx,)
            that contains the standard deviation of all features of X

    Returns:
        numpy.ndarray: Normalized X matrix
    """
    return (X - m) / s
