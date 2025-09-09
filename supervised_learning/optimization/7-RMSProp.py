#!/usr/bin/env python3
"""Met à jour une variable en utilisant l'algorithme RMSProp."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Met à jour une variable en utilisant l'algorithme RMSProp.

    Args:
        alpha (float): taux d'apprentissage.
        beta2 (float): poids RMSProp (RMSProp weight).
        epsilon (_type_): small number to avoid divison by zero
        var (numpy.ndarray): variable à mettre à jour, de forme (784, 1). W
        grad (numpy.ndarray): gradient de la variable, de forme (784, 1). dW
        s (numpy.ndarray): précédent second moment de var. VdW

    Returns:
        tuple: la variable mise à jour et le nouveau moment
    """
    # Calcul des moyennes exponentiellement pondérées
    s = beta2 * s + (1 - beta2) * grad**2
    # Mise à jour des poids ou des biais
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
