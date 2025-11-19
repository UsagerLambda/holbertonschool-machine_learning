#!/usr/bin/env python3
"""Function that calculates the determinant of a matrix."""


def inverse(matrix):
    """Calculate the inverse of a matrix."""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    adj = adjugate(matrix)
    if det == 0:
        return None
    B = []
    for i in range(len(adj)):
        new_row = []
        for j in range(len(adj)):
            new_row.append(adj[i][j] / det)
        B.append(new_row)
    return B


def adjugate(matrix):
    """Calculate the adjugate matrix of a matrix."""
    cof = cofactor(matrix)
    n = len(cof)
    if n == 1:
        cof[0][0] = 1
        return cof

    for i in range(n):
        for j in range(i + 1, n):
            cof[i][j], cof[j][i] = cof[j][i], cof[i][j]

    return cof


def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix."""
    result = minor(matrix)
    n = len(result)

    for i in range(n):
        for j in range(n):
            result[i][j] *= (-1) ** (i + j)

    return result


def minor(matrix):
    """Calculate the minor matrix of a matrix."""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []

    for i in range(n):  # Parcourt les lignes
        row = []
        for j in range(n):
            # Créer une sous-matrice en supprimant la ligne i et la colonne j
            sub_matrix = []  # Parcourt les colonnes
            for row_idx in range(n):  # Parcourt les lignes (de nouveau)
                if row_idx == i:  # Si même ligne que le minor recherché
                    continue  # skip
                new_row = []
                for col_idx in range(n):
                    if col_idx == j:  # Si même colonne que minor recherché
                        continue  # skip
                    # Enregistre les indexs de la lignes pour la sub matrice
                    new_row.append(matrix[row_idx][col_idx])
                sub_matrix.append(new_row)  # Ajoute la ligne d'index
            # Calculer le déterminant de cette sous-matrice
            det = determinant(sub_matrix)  # Calcule
            row.append(det)  # Enregistre les det de la ligne
        minor_matrix.append(row)  # Enregistre la ligne

    return minor_matrix


def determinant(matrix):
    """Calculate the determinant of a matrix."""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        return 1

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
