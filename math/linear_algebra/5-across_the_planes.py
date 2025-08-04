#!/usr/bin/env python3
"""
Function that return the sum of two matrice element-wise
"""


def add_matrices2D(mat1, mat2):
    """add_matrice2D

    Args:
        mat1 (list of lists 2D): 2D matrice containing ints
        mat2 (list of lists 2D): 2D matrice containing ints

    Returns:
        list of lists: sum of two matrices element-wise
    """

    if len(mat1) != len(mat2):  # Vérifie le nombre de lignes
        return None

    for i in range(len(mat1)):  # Vérifie le nombre de colonnes
        if len(mat1[i]) != len(mat2[i]):
            return None

    new = []
    for i in range(len(mat1)):  # Boucle dans les lignes
        temp = []  # Reinitialise temp
        for y in range(len(mat1[0])):  # Boucle dans les colonnes
            temp.append(mat1[i][y] + mat2[i][y])
        new.append(temp)  # Ajoute temp dans new
    return new
