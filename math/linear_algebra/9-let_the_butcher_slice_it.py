#!/usr/bin/env python3
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 = matrix[1:3, :]  # Lignes 1 et 2, toutes les colonnes
mat2 = matrix[:, 2:4]  # Toutes les lignes, colonnes 2 et 3
mat3 = matrix[1:4, 3:6]  # Lignes 1 à 3, colonnes 3 à 5
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
