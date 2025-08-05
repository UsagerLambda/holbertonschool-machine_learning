#!/usr/bin/env python3
"""
Function that add mat1 and mat2 element-wise
"""


def add_matrices(mat1, mat2):
    """add_matrice

    Args:
        mat1 (list): A matrix of a possibly nested list of lists
        mat2 (list): A matrix of a possibly nested list of lists

    Returns:
        list or None: A new matrix resulting from the element-wise
        addition of `mat1` and `mat2`, or None if the two matrices
        have different shapes.
    """
    if shape(mat1) != shape(mat2):
        return None

    new = add(mat1, mat2)
    return new


def add(mat1, mat2):
    """add
    Recursively add two matrices (nested lists) element-wise.

    Args:
        mat1 (list): A matrix of a possibly nested list of lists
        mat2 (list): A matrix of a possibly nested list of lists

    Returns:
        list: A new matrix or list containing the element-wise
        sum of `mat1` and `mat2`.
    """
    result = []
    for e1, e2 in zip(mat1, mat2):
        if isinstance(e1, list) and isinstance(e2, list):
            result.append(add(e1, e2))  # recursion magic
        else:
            result.append(e1 + e2)
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
