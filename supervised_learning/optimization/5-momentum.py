#!/usr/bin/env python3
"""Update weights or bias and calculate the moment."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update var using the gradient descend with momentum optimization.

    Args:
        alpha (float): taux d'apprentissage.
        beta1 (float): coefficient du momentum.
        var (numpy.ndarray): variable à mettre à jour, de forme (784, 1). W
        grad (numpy.ndarray): gradient de la variable, de forme (784, 1). dW
        v (numpy.ndarray): précédent premier moment de la variable, VdW
            de forme (784, 1).

        tuple: la variable mise à jour et le nouveau moment, respectivement.
    """
    # Calcul des moyennes exponentiellement pondérées
    v = beta1 * v + (1 - beta1) * grad
    # Mise à jour des poids ou des biais
    var = var - alpha * v
    return var, v
