#!/usr/bin/env python3

def determinant(matrix):
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # 1x1 (det == unique argument)
    if n == 1:
        return matrix[0][0]

    # 2x2
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):  # Parcourt chaque colonne de la première ligne
        minor = []
        for i in range(1, n):  # Parcourt chaque ligne sauf la première
            row = []
            for k in range(n):  # Parcourt chaque colonne
                if k != j:  # Ignore la colonne courante
                    row.append(matrix[i][k])
            minor.append(row)  # Ajoute la ligne réduite au mineur
        # Calcule le cofacteur correspondant à l'élément (0, j)
        cofactor = ((-1) ** j) * matrix[0][j] * determinant(minor)
        det += cofactor

    return det
