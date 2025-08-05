#!/usr/bin/env python3
"""
Function that add, substract, multiply and divide mat1 with mat2
"""


def np_elementwise(mat1, mat2):
    """np_elementwise

    Args:
        mat1 (matrice): matrice of integers or float
        mat2 (matrice): matrice of integers or float

    Returns:
        tuple: result of adding, substracting, multiply and divide mat1 & mat2
    """
    new = (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
    return new
