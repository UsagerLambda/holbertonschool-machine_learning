#!/usr/bin/env python3
"""Module qui définit un réseau de neurones profond."""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Define a deep neural network performing binary classification.

        Args:
            nx (int): number of input features
            layers (list): list representing the number if nodes
            in each layer of the network

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is not a positive integer
            TypeError: if layers is not a list, is empty or elements
            in the list are not integer or not positive
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        # --- Private instance attributes ---

        # Chaque entier de la liste est le nombre de neurones par couche,
        # la longueur de la liste est le nombre de couches
        self.__L = len(layers)

        # Dictionnaire qui stocke les valeurs du calcul linéaire
        # et de l'activation de chaque couches
        self.__cache = {}

        self.__weights = {}
        for index in range(len(layers)):
            if not isinstance(layers[index], int) or layers[index] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Si c'est la première couche
            if index == 0:
                # Utilisation de la méthode He et al.
                # pour réajuster les poids pour éviter la
                # saturation des activations
                self.weights[f"W{index+1}"] = np.random.randn(
                    layers[index], nx) * np.sqrt(2 / nx)
            else:  # Sinon
                self.weights[f"W{index+1}"] = np.random.randn(
                    layers[index], layers[index-1]) * np.sqrt(
                        2 / layers[index-1])

            # Initialise le biais de la couche l+1
            # avec un tableau colonne de zéros,
            # une valeur par neurone de la couche
            self.weights[f"b{index+1}"] = np.zeros((layers[index], 1))

    @property
    def L(self):
        """Retourne la valeur de self.__L."""
        return self.__L

    @property
    def cache(self):
        """Retourne la valeur de self.__cache."""
        return self.__cache

    @property
    def weights(self):
        """Retourne la valeur de self.__weights."""
        return self.__weights
