#!/usr/bin/env python3
"""
Function that performs matrix multiplication by doing the
multiplication of the index line of mat1 with columns index of mat2
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

    for i in range(len(mat1)):  # Parcourt les lignes de mat1
        ligne = []  # Reinitialise lignes
        for y in range(len(mat2[0])):  # Parcourt les colonnes de mat2
            # Calcule horrible
            val = (mat1[i][0] * mat2[0][y] + mat1[i][1] * mat2[1][y])
            # print(f"calcul. mat1[{i}][0] * mat2[0][{y}]
            #              + mat1[{i}][1] * mat2[1][{y}]")
            # print(f"1. {mat1[i][0]} * {mat2[0][y]}
            #               + {mat1[i][1]} * {mat2[1][y]}")
            # print(f"2. {mat1[i][0] * mat2[0][y]}
            #               + {mat1[i][1] * mat2[1][y]}")
            # print(f"3. {mat1[i][0] * mat2[0][y]
            #                + mat1[i][1] * mat2[1][y]}")
            ligne.append(val)  # ajoute les valeurs dans le tableau de ligne
        new.append(ligne)  # Ajoute la ligne au tableau final
    return new
