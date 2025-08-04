#!/usr/bin/env python3

def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):  # Tant que matrix est une liste
        shape.append(len(matrix))  # Ajoute la taille de matrix
        matrix = matrix[0]  # matrix "descend" dans la liste imbiquée
    return shape
