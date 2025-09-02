#!/usr/bin/env python3
"""Module qui définit un réseau de neurones profond."""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Define a deep neural network performing binary classification.

        Args:
            nx (int): number of input features
            layers (list): list representing the number if nodes
                in each layer of the network
            activation (str): define what activation method to use
                in the forward function.

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

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        # --- Private instance attributes ---

        # Chaque entier de la liste est le nombre de neurones par couche,
        # la longueur de la liste est le nombre de couches
        self.__L = len(layers)

        # Dictionnaire qui stocke les valeurs du calcul linéaire
        # et de l'activation de chaque couches
        self.__cache = {}

        self.__activation = activation

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

    @property
    def activation(self):
        """Retourne la valeur self.__activation."""
        return self.__activation

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

            Using the sigmoid function or tanh function based on
            self.__activation.

        The method computes activations layer by layer:
            - For hidden layers (1 to L-1), the sigmoid activation is applied.
            - For the output layer (L), the softmax activation is applied,
            producing a probability distribution over the classes.

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
            Z = np.matmul(self.__weights[f"W{i+1}"], self.__cache[
                    f"A{i}"]) + self.__weights[f"b{i+1}"]  # Calcul les logits
            if i == self.__L - 1:
                # Soustrait le max de Z,
                # pour éviter valeurs exponentielles trop grandes
                softMaxSub = Z - np.max(Z, axis=0, keepdims=True)
                # Exponentielle pour chaque valeurs
                softMaxExp = np.exp(softMaxSub)
                # Calcul la sommes des exp (pour normalisation)
                softMaxSum = np.sum(softMaxExp, axis=0, keepdims=True)
                # Calcul de normalisation (exp / sum(exp))
                softMax = softMaxExp / softMaxSum
                self.__cache[f"A{i+1}"] = softMax

        # --- sigmoid ---
            if i < self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache[f"A{i+1}"] = (1 / (1 + np.exp(-Z)))
                if self.__activation == 'tanh':
                    self.__cache[f"A{i+1}"] = np.tanh(Z)

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
        J = -(1/m * np.sum(Y * np.log(A + 0)))
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

        # indice des valeurs les plus élevées dans les colonnes
        max = np.argmax(A, axis=0)
        nb_classes = A.shape[0]  # Récupère le nombre de classes
        # Créer un tableau one-hot, où chaque colonnes indique
        # la classe prédite pour chaque exemple
        predictions = np.eye(nb_classes)[max].T

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
        DZ = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache[f"A{i-1}"]

            DW = np.matmul(DZ, A_prev.T) / m
            DB = np.sum(DZ, axis=1, keepdims=True) / m

            if i > 1:
                if self.__activation == 'sig':
                    derivative = A_prev * (1 - A_prev)
                if self.__activation == 'tanh':
                    derivative = 1 - A_prev ** 2

                DZ = np.matmul(self.__weights[f"W{i}"].T, DZ) * derivative

            self.__weights[f"W{i}"] = self.__weights[f"W{i}"] - alpha * DW
            self.__weights[f"b{i}"] = self.__weights[f"b{i}"] - alpha * DB

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Entraîne le réseau de neurones profond.

        Args:
            X (np.ndarray): Données d'entrée de forme (nx, m)
            Y (np.ndarray): Étiquettes réelles de forme (1, m)
            iterations (int, optional): Nombre d'itérations. Defaults to 5000.
            alpha (float, optional): Taux d'apprentissage. Defaults to 0.05.

        Raises:
            TypeError: si iterations n'est pas un int
            ValueError: si iterations < 1
            TypeError: si alpha n'est pas un float
            ValueError: si alpha <= 0

        Returns:
            tuple: (prédictions finales, coût final)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # --- Si graph = True, initialise deux variables pour stocker,
        # les coûts et la liste d'iterations ---
        if graph:
            costs = []
            iterations_list = []

        # --- Boucle d’entraînement ---
        for i in range(iterations + 1):
            _, cache = self.forward_prop(X)
            cost = self.cost(Y, cache[f"A{self.__L}"])

            # --- si graph = True,
            # stocke tous les multiple de step + last iteration ---
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                iterations_list.append(i)

            # --- si verbose = True, alors print le coût si l'iteration
            # est un multiple de step ou la dernière iteration ---
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # --- Si graph = True, affiche le graph
        if graph:
            plt.plot(iterations_list, costs, color='blue', linestyle='-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # --- Évaluation finale ---
        predictions, J = self.evaluate(X, Y)
        return predictions, J

    def save(self, filename):
        """
        Save the instance object into a file in pickle format.

        Args:
            filename (str): filename
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load object from a pickle file.

        Args:
            filename (str): name of the file to load

        Returns:
            object: loaded object or None if filename doesn't exist.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
