#!/usr/bin/env python3
"""Module de politique par gradient pour l'apprentissage par renforcement."""

import numpy as np


def policy(matrix, weight):
    """Calculate la politique stochastique via softmax sur les scores pondérés.

    Args:
        matrix: numpy.ndarray de forme (n,) représentant l'état courant.
        weight: numpy.ndarray de forme (n, m) représentant la matrice de
            poids, où m est le nombre d'actions possibles.

    Returns:
        numpy.ndarray de forme (m,) contenant la distribution de probabilité
        sur les actions.
    """
    # state * weight donne un score par action
    z = np.dot(matrix, weight)
    # Rend positif + emplifie les écarts
    exp_z = np.exp(z)
    # Divise par le total -> somme à 1 -> proba
    probs = exp_z / np.sum(exp_z)
    return probs


def policy_gradient(state, weight):
    """Calculate l'action choisie et le gradient de la politique.

    Args:
        state: numpy.ndarray de forme (1, n) représentant l'état courant.
        weight: numpy.ndarray de forme (n, m) représentant la matrice de
            poids, où m est le nombre d'actions possibles.

    Returns:
        tuple: (action, gradient) où action est l'indice de l'action
            choisie et gradient est le numpy.ndarray de forme (n, m)
            représentant le gradient de log-probabilité de l'action.
    """
    # Probabilités des actions selon la politique softmax
    probs = policy(state, weight)
    # Échantillonne une action selon la distribution de probabilité
    action = np.random.choice(len(probs), p=probs)
    # Encodage one-hot de l'action choisie
    one_hot = np.zeros_like(probs)
    one_hot[action] = 1
    # Gradient = outer(state, one_hot - probs) : dérivée log softmax -> (n, m)
    gradient = np.outer(state, one_hot - probs)

    return action, gradient
