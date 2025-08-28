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

    def forward_prop(self, X):
        """Compute forward propagation for a 2-layer neural network.

        Each layer receives as input the activation values of the
        previous layer (A[l-1]):
            - The first layer receives the raw input X (A0 = X).
            - Hidden layer: Z1 = W1·X + b1, A1 = sigmoid(Z1).
            - Output layer: Z2 = W2·A1 + b2, A2 = sigmoid(Z2).

        Args:
            X (np.ndarray): input data with shape (nx, m)
                - nx: number of features per example.
                - m: number of examples.

        Returns:
            tuple: (A1, A2)
                - A1 (np.ndarray): Activations from the hidden layer.
                - A2 (np.ndarray): Activations from the output layer.
        """
        # --- Couche cachée ---
        # Calcul linéaire : Z1 = W1 · X + b1
        Z1 = np.dot(self.__W1, X) + self.__b1
        # Activation sigmoïde
        self.__A1 = (1 / (1 + np.exp(-Z1)))

        # --- Couche de sortie ---
        # Reçois l'activation de la couche cachée.
        # # Calcul linéaire : Z2 = W2 · A1 + b2
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        # Activation sigmoïde
        self.__A2 = (1 / (1 + np.exp(-Z2)))

        return self.__A1, self.__A2

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

        Le neurone calcule d'abord la sortie activée A2 via forward_prop,
        puis :
            - Calcule le coût moyen (loss) en comparant A2 à Y.
            - Convertit A2 en prédictions binaires (1 si ≥ 0.5, sinon 0).

        Args:
            X (np.ndarray): Données d'entrée de forme (nx, m)
            Y (np.ndarray): Étiquettes réelles de forme (1, m)

        Returns:
            tuple: (predictions (np.ndarray), coût (float))
        """
        # --- Étape 1 : propagation avant ---
        # On ne garde que la sortie finale A2 (prédictions du modèle)
        _, A2 = self.forward_prop(X)  # car forward_prop renvoie A1 & A2

        # --- Étape 2 : calcul du coût ---
        # On compare les sorties du modèle A2 avec les vraies étiquettes Y
        J = self.cost(Y, A2)

        # --- Étape 3 : génération des prédiction binaires ---
        # Crée un array de 0 et 1 selon si chaque élément de A est <0.5 ou ≥0.5
        predictions = np.where(A2 >= 0.5, 1, 0)

        return predictions, J

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Itération de descente de gradient pour ajuster les poids et biais.

        Cette fonction met à jour :
            - les poids et biais de la couche de sortie (W2, b2)
            - les poids et biais de la couche cachée (W1, b1).

        Args:
        X (np.ndarray): données d'entrée de forme (nx, m),
            nx = nombre de features, m = nombre d'exemples.
        Y (np.ndarray): étiquettes réelles de forme (1, m).
        A1 (np.ndarray): activations de la couche cachée de forme (nodes, m).
        A2 (np.ndarray): activations de la couche de sortie de forme (1, m).
        alpha (float, optional): taux d'apprentissage. Contrôle la taille
            des corrections de poids. Defaults to 0.05.
        """
        # Nombre d'exemples dans notre dataset (combien de lignes on traite)
        m = Y.shape[1]

        # ---- Gradient descent du neuron de sortie ----
        # Ècart entre l'activation du neurone de sortie et l'étiquette
        dz2 = A2 - Y

        # dw2 stocke le gradient moyen de chaque poids de la couche de sortie
        # par rapport à l'erreur de sortie des cette couche (Écart)
        # pour tous les exemples
        dw2 = np.dot(dz2, A1.T) / m

        # db2 = gradient des biais de la couche de sortie
        # Correspond à la moyenne des erreurs dz2
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # ---- Gradient descent + Back propagation de l'hidden output ----

        # "np.dot(self.__W2.T, dz2)"
        # transmet l’erreur de sortie à la couche cachée selon les poids

        # "* (A1 * (1 - A1))"
        # ajuste l'erreur par rapport à la sensibilité du neurone caché
        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))

        # dw1 stocke le gradient moyen de chaque poids de la couche cachée
        # par rapport à l'erreur de sortie de cette couche (dz1)
        # pour tous les exemples.
        dw1 = np.dot(dz1, X.T) / m

        # db1 = gradient des biais de la couche cachée
        # Correspond à la moyenne des erreurs dz1
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # --- mise à jour des variables ---

        # Corrige les poids : si dw2 dit "trop fort", je diminue (d'où le -)
        # alpha contrôle si : petite correction (0.01) ou grosse (0.5)
        self.__W2 = self.__W2 - alpha * dw2

        # Correction du biais selon db2
        # On ajuste le biais du neurone en fonction de la moyenne
        # des erreurs (db2). Cela permet au neurone de s'adapter
        # globalement à la tendance des prédictions.
        self.__b2 = self.__b2 - alpha * db2

        # Corrige les poids : si dw1 dit "trop fort", je diminue (d'où le -)
        # alpha contrôle si : petite correction (0.01) ou grosse (0.5)
        self.__W1 = self.__W1 - alpha * dw1

        # Correction du biais selon db1
        # On ajuste le biais du neurone en fonction de la moyenne
        # des erreurs (db1). Cela permet au neurone de s'adapter
        # globalement à la tendance des prédictions.
        self.__b1 = self.__b1 - alpha * db1
