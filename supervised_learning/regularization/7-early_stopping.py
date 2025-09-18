#!/usr/bin/env python3
"""Module pour l'arrêt précoce dans les réseaux de neurones."""

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Détermine si l'arrêt précoce doit être appliqué.

    L'arrêt précoce doit se produire lorsque le coût de validation du réseau
    n'a pas diminué par rapport au coût de validation optimal de plus que le
    seuil sur un nombre spécifique de patience.

    Args:
        cost (float): Le coût de validation actuel du réseau de neurones
        opt_cost (float): Le coût de validation le plus bas enregistré du
            réseau de neurones
        threshold (float): Le seuil utilisé pour l'arrêt précoce
        patience (int): Le nombre de patience utilisé pour l'arrêt précoce
        count (int): Le nombre de fois que le seuil n'a pas été atteint

    Returns:
        tuple: Un booléen indiquant si le réseau doit être arrêté
            prématurément, suivi du nombre mis à jour
    """
    if (cost < opt_cost - threshold):
        opt_cost = cost
        count = 0
    else:
        count += 1
        if (count >= patience):
            return True, count
    return False, count
