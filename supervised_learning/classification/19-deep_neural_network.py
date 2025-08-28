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

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m), where
                                nx is the number of input features
                                and m is the number of examples.

        Returns:
            A (numpy.ndarray): The output of the neural network after
                                forward propagation.
            cache (dict): A dictionnary containing all the activated
                                outputs of each layer, with the key "A0"
                                for the input data and "A{i}" for the layer i.
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            Z = np.dot(self.__weights[f"W{i+1}"], self.__cache[
                f"A{i}"]) + self.__weights[f"b{i+1}"]
            self.__cache[f"A{i+1}"] = (1 / (1 + np.exp(-Z)))

        return self.__cache[f"A{i+1}"], self.__cache

    def cost(self, Y, A):
        """Calculate le coût du modèle en régression logistique.

        Args:
            Y (numpy.ndarray): Tableau de forme (1, m),
            contenant les vraies étiquettes.
            A (numpy.ndarray): Tableau de forme (1, m),
            contenant les sorties activées du neurone.

        Returns:
            float: Le coût moyen (loss) du modèle sur les m exemples.
        """
        m = Y.shape[1]  # nombre d'éléments dans les colonnes du tableau numpy
        J = -(1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        return J
