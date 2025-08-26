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

        # --- Private instance attributes pour couches cachées ---
        self.__W1 = np.random.randn(nodes, nx)
        # Vecteur de biais de la couche cachée,
        # ayant une dimension pour chaque neurone de celle-ci (initialisé à 0)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # --- Private instance attributes pour neurons de sortie ---
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # --- getter ---

    @property
    def W1(self):
        """Retourne la valeur de self.__W1."""
        return self.__W1

    @property
    def b1(self):
        """Retourne la valeur de self.__b1."""
        return self.__b1

    @property
    def A1(self):
        """Retourne la valeur de self.__A1."""
        return self.__A1

    @property
    def W2(self):
        """Retourne la valeur de self.__W2."""
        return self.__W2

    @property
    def b2(self):
        """Retourne la valeur de self.__b2."""
        return self.__b2

    @property
    def A2(self):
        """Retourne la valeur de self.__A2."""
        return self.__A2
