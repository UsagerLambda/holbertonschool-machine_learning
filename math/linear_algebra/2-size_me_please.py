#!/usr/bin/env python3
"""
Ce module contient la fonction matrix_shape qui calcule les dimensions
d'une matrice potentiellement 1D, 2D ou 3D représentée par une liste imbriquée.
"""

def matrix_shape(matrix):
    """matrix_shape

    Calcule les dimensions d'une matrice (liste imbriquée).
    Liste les dimensions de la matrice dans l'ordre suivant:
        Si 3D : nombre de blocs (Z axis), nombre de lignes (Y axis),
            nombre de colonnes (X axis)
        Si 2D : nombre de lignes (Y axis), nombre de colonnes (X axis)

    Args:
        matrix (list): Liste imbriquée 1D, 2D ou 3D...

    Returns:
        list: taille de la matrice donnée
    """
    shape = []
    while isinstance(matrix, list):  # Tant que matrix est une liste
        shape.append(len(matrix))  # Ajoute la taille de matrix
        matrix = matrix[0]  # matrix "descend" dans la liste imbiquée
    return shape
