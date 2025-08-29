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

    def evaluate(self, X, Y):
        """Évalue le neurone et retourne les prédictions et le coût.

        Cette méthode effectue les opérations suivantes :
        1. Calcule l'activation de sortie A via forward propagation.
        2. Calcule le coût moyen entre les prédictions A
            et les vraies étiquettes Y.
        3. Convertit les activations A en prédictions binaires :
           1 si A ≥ 0.5, sinon 0.

        Args:
            X (np.ndarray): Données d'entrée de forme (nx, m)
            Y (np.ndarray): Étiquettes réelles de forme (1, m)

        Returns:
            tuple: (predictions (np.ndarray), coût (float))
        """
        # car forward_prop renvoie l'activation de la couche output & le cache
        A, _ = self.forward_prop(X)
        J = self.cost(Y, A)

        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, J

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one iteration of gradient descent on the neural network.

        Update the weights and biases of each layer using backpropagation.
        For each layer (starting from the last):
            1. Compute the layer's error (DZ).
            2. Compute the gradients of the weights (DW) and biases (DB).
            3. Update the weights and biases by subtracting
                alpha times the gradients.

        Args:
            Y (np.ndarray): Array of true labels,
                shape (1, m), where m is the number of examples.
            cache (dict): Dictionary containing the activations of each layer
                during forward propagation.
                      Keys are "A0", "A1", ..., "AL".
            alpha (float, optional): Learning rate used to update parameters.
                Defaults to 0.05.
        """
        m = Y.shape[1]
        save = None

        for i in range(self.__L, 0, -1):
            if i == self.__L:
                DZ = cache[f"A{i}"] - Y
            else:
                DZ = np.dot(self.__weights[f"W{i+1}"].T, save) * (
                    cache[f"A{i}"] * (1 - cache[f"A{i}"]))

            DW = np.dot(DZ, cache[f"A{i-1}"].T) / m
            DB = np.sum(DZ, axis=1, keepdims=True) / m

            self.__weights[f"W{i}"] = self.__weights[f"W{i}"] - alpha * DW
            self.__weights[f"b{i}"] = self.__weights[f"b{i}"] - alpha * DB

            save = DZ
