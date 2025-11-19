#!/usr/bin/env python3
"""Multivariate Normal distribution."""

import numpy as np


class MultiNormal:
    """Multivariate Normal distribution."""

    def __init__(self, data):
        """Class constructor of a Multivariate Normal distribution."""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculate the PDF at a data point."""
        pass
