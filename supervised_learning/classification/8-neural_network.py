#!/usr/bin/env python3
"""Module qui définit un réseau de neurones."""

import numpy as np


class NeuralNetwork:
    """Classe réseau de neurones à une couche cachée, binaire."""

    def __init__(self, nx, nodes):
        """
        Neural network with one hidden layer, binary classification.

        Args:
            nx (int): number of input features
            nodes (int): number of neurons in the hidden layer

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is not positive
            TypeError: if nodes is not an integer
            ValueError: if nodes is not positive
        """
        # --- Validation des parmètres ---
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # --- Public instance attributes pour couches cachées ---
        self.W1 = np.random.randn(nodes, nx)
        # Vecteur de biais de la couche cachée,
        # ayant une dimension pour chaque neurone de celle-ci (initialisé à 0)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # --- Public instance attributes pour neurons de sortie ---
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
