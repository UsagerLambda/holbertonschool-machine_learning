#!/usr/bin/env python3
"""
Ce module contient une fonction permettant de transposer une matrice 2D.
La transposition d'une matrice consiste à permuter ses lignes et ses colonnes.
"""


def matrix_transpose(matrix):
    """matrix_transpose

    La méthode matrix_transpose échange les lignes et les colonnes.
    exemple: matrix = [[1, 2], [3, 4], [5, 6]] deviens [[1, 3, 5], [2, 4, 6]]

    Args:
        matrix (list of lists): Matrice 2D à transposer

    Returns:
        list of lists: Nouvelle matrice transposée
    """
    transposed = []  # Liste qui contiendra la matrice transposée
    temp = []  # Liste temporaire

    for i in range(len(matrix[0])):  # each indice of colonnes dans une ligne
        for lignes in matrix:  # Pour chaque ligne dans matrix
            temp.append(lignes[i])  # Add l'élément of  colonne i to list temp
        transposed.append(temp)  # Ajoute la ligne temp dans la list transposed
        temp = []  # Réinitialise la variable temp

    return transposed  # Retourne la liste transposed
