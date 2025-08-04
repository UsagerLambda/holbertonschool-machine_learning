#!/usr/bin/env python3
"""
Function that performs matrix multiplication by doing the
multiplication of the index line of mat1 with columns index of mat2

exemple:
mat1 = [[1, 2],
        [3, 4]]

mat2 = [[5, 6],
        [7, 8]]
>>> [[19, 22],  -> (1*5 + 2*7), (1*6 + 2*8)
>>> [43, 50]]   -> (3*5 + 4*7), (3*6 + 4*8)
"""


def mat_mul(mat1, mat2):
    """mat_mul

    Args:
        mat1 (list of lists): list of lists of ints
        mat2 (lists of lists): list of lists of ints

    Returns:
        list: list of lists of multiplied ints
    """
    new = []

    if len(mat1[0]) != len(mat2):
        return None

    for i in range(len(mat1)):  # Parcourt les lignes de mat1
        ligne = []  # Reinitialise lignes
        for y in range(len(mat2[0])):  # Parcourt les colonnes de mat2
            val = 0  # Reinitialise val
            for k in range(len(mat2)):  # Parcourt les lignes de mat2
                # print(f"val{k} += mat1[{i}][{k}] * mat2[{k}][{y}]
                #           or val += {mat1[i][k]} * {mat2[k][y]}")
                # print(f"val calcul {val} + {mat1[i][k] * mat2[k][y]}")
                val += mat1[i][k] * mat2[k][y]
                # print(f"val = {val}")
            # print(f"result = {val}")
            ligne.append(val)
        new.append(ligne)
    return new
