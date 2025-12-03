#!/usr/bin/env python3
"""Calculate the probability density function of a guassian distribution."""

import numpy as np


def pdf(X, m, S):
    """
    Calculate the probability density function of a guassian distribution.

    Calcule la proba qu'un point appartienne à un cluster donné.

    PDF(x | μ, Σ) = (1 / √((2π)^d × |Σ|)) × exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))

    Args:
        X (np.ndarray): of shape (n, d)
            containing the data points whose PDF should be evaluated
        m (np.ndarray): of shape (d,)
            containing the mean of the distribution
        S (np.ndarray): of shape (d, d)
            containing the covariance of the distribution

    Return:
        P or None on failure:
            P is a np.ndarray of shape (n,)
                containing the PDF values for each data point
    """
    if not isinstance(
        X, np.ndarray) or not isinstance(
            m, np.ndarray) or not isinstance(
                S, np.ndarray):
        return None

    if X.ndim != 2 or m.ndim != 1 or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[
        0] or S.shape[
            0] != S.shape[
                1] or S.shape[
                    0] != m.shape[0]:
        return None

    d = m.shape[0]  # Nombre de dimension

    coeff = (2 * np.pi) ** d * np.linalg.det(S)
    norm = 1 / np.sqrt(coeff)

    diff = X - m
    S_inv = np.linalg.inv(S)
    exp = np.sum(diff @ S_inv * diff, axis=1)
    exp_p = np.exp(-0.5 * exp)

    return np.maximum(norm * exp_p, 1e-300)
