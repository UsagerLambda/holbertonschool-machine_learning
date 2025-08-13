#!/usr/bin/env python3
"""Module pour calculer la dérivée d'un polynôme représenté par une liste."""


def poly_derivative(poly):
    """
    Calcule la dérivée d'un polynôme.

    Args:
        poly (list): Liste des coefficients du polynôme (ordre croissant).

    Returns:
        list: Liste des coefficients de la dérivée, ou None si entrée invalide.
    """
    if not isinstance(poly, list):
        return None

    if not all(isinstance(k, int) for k in poly):
        return None

    result = []
    for i in range(1, len(poly)):
        result.append(i * poly[i])

    if not result:
        return None

    return result
