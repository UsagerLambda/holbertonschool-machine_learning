#!/usr/bin/env python3
"""Update var using Adam optimization."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable in place using the Adam optimization.

    Args:
        alpha (float): taux d'apprentissage.
        beta1 (float): weight used for the first moment
        beta2 (float): weight used for the second moment
        epsilon (float): small number to avoid division by zero
        var (numpy.ndarray): containing the variable to be updated (W)
        grad (numpy.ndarray): containing the gradient of var (dW)
        v (numpy.ndarray): previous first moment of var (VdW)
        s (numpy.ndarray): previous second moment of var (VdW)
        t (int): time step used for bias correction

    Returns:
        tuple: The updated var, the new first and second moment
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_corrected = v / (1 - beta1**t)
    s_corrected = s / (1 - beta2**t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
