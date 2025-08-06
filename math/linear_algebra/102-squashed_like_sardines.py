#!/usr/bin/env python3
"""
Function qui concatène deux matrices ayant la même forme
"""


def cat_matrices(mat1, mat2, axis=0):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if len(shape(mat1)) != len(shape(mat2)):
        return None

    """
    Si deux matrices ont la même taille sur toutes les dimensions sauf celle
    spécifiée par l'axis, alors elles peuvent être concaténées sur cet axis.
    """
    for i in range(len(shape(mat1))):
        # Saute l'axe de concaténation
        if i != axis:
            # Si les autres dimensions diffèrent
            if shape(mat1)[i] != shape(mat2)[i]:
                return None

    return concatenate(mat1, mat2, axis)


def concatenate(mat1, mat2, axis):
    """concatenate

    Args:
        mat1 (list): Matrice sous forme de listes imbirquées
        mat2 (list): Matrice de même forme que mat1
        axis (int): Axe le long duquel concaténer

    Returns:
        list: Nouvelle matrice résultant
        de la concaténation de mat1 & mat2 sur l'axe donné
    """
    # Si axis == 0, alors on est au bon niveau de profondeur
    if axis == 0:
        return mat1 + mat2
    result = []
    # Parcourt mat1 & mat2 en parallèle
    for e1, e2 in zip(mat1, mat2):
        # Recursion pour descendre dans les niveaux jusqu'à axis == 0
        concatenated = concatenate(e1, e2, axis - 1)
        # Lors de la remontée, ajoute le résultat concaténé de ce sous-niveau
        # à la liste result de ce niveau supérieur
        result.append(concatenated)
    # Une fois la récursion fini, retourne la liste complète
    return result


def shape(mat):
    """shape
    Determine the shape of the given matrix

    Args:
        mat (list): A matrix represented as a nested lists.

    Returns:
        list: List of integer representing the shape of the matrix
    """
    shape = []
    while isinstance(mat, list):
        shape.append(len(mat))
        mat = mat[0]
    return shape
