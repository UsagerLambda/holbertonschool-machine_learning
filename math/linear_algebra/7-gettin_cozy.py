#!/usr/bin/env python3
"""
Return a concatenates list, extended via a specified axis.
0 for ligne, 1 for columns
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cat_matrice2D

    Args:
        mat1 (list de lists): list de lists
        mat2 (list de lists): list de lists
        axis (int, optional): define the axis how the two matrices
        will be concatenates.
        Defaults to 0.

    Returns:
        list_: list of lists
    """
    new = []

    if axis == 0:
        if len(mat2[0]) != len(mat1[0]):
            return None

        for ligne in mat1:
            new.append(ligne.copy())

        new.extend(mat2)
        return new

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        for ligne in mat1:
            new.append(ligne.copy())

        for i in range(len(new)):
            new[i].extend(mat2[i])
        return new
