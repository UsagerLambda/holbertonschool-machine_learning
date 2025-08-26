#!/usr/bin/env python3
"""Module qui définit la classe Neuron pour un réseau de neurones simple."""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Classe d'un neurone pour un réseau à une couche."""

    def __init__(self, nx):
        """
        Initialise un neurone.

        Paramètres :
            nx (int) : Nombre de caractéristiques d'entrée du neurone.

        Lève :
            TypeError : Si nx n'est pas un entier.
            ValueError : Si nx est inférieur à 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx  # Nombre de feature du neurone

        # Créer un tableau numpy de 1 ligne * nx colonnes
        # vecteur de poids aléatoires autour de 0
        # (valeurs positives et négatives, la plupart proches de 0)
        # En gros ça créer des valeurs de poids pour chaque features
        self.__W = np.random.randn(1, nx)
        # Set le biais sur 0
        self.__b = 0
        # Résultat de la sortie calculée par le neurone. (0 de base)
        self.__A = 0

    # Getter pour permettre la lecture des données via
    # l'exterieur du fichier mais pas la modification
    @property
    def W(self):
        """Retourne la valeur de self.__W."""
        return self.__W

    @property
    def b(self):
        """Retourne la valeur de self.__b."""
        return self.__b

    @property
    def A(self):
        """Retourne la valeur de self.__A."""
        return self.__A

    def forward_prop(self, X):
        """Fonction qui performe une classification binaire.

        Le neurone utilise une fonction d'activation sigmoïde.

        Args:
            X (np.array): numpy array avec la forme (nx, m),
            qui contient les données entrantes.

        Returns:
            private_attribute: retourne la valeur à l'attribut privé __A.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = (1 / (1 + np.exp(-Z)))
        return self.__A

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

        Le neurone calcule d'abord la sortie activée A via forward propagation,
        convertit A en prédictions binaires (1 si >=0.5, sinon 0), puis calcule
        le coût moyen sur tous les exemples.

        Args:
            X (np.ndarray): Données d'entrée de forme (nx, m)
            Y (np.ndarray): Étiquettes réelles de forme (1, m)

        Returns:
            tuple: (predictions (np.ndarray), coût (float))
        """
        A = self.forward_prop(X)  # Calcule la sortie activée
        J = self.cost(Y, A)  # Calcule du coût
        # Crée un array de 0 et 1 selon si chaque élément de A est <0.5 ou ≥0.5
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, J

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Effectue une itération de descente de gradient.

        Args:
            X (np.ndarray): Doonées d'entrée de forme (nx, m)
            Y (np.ndarray): Lables corrects de forme (1, m)
            A (np.ndarray): Sorties activées de forme (1, m)
            alpha (float, optional): Taux d'appentissage. Defaults to 0.05.

        Met à jour les poids __W et le biais __b.
        """
        # Nombre d'exemples dans notre dataset (combien de lignes on traite)
        m = Y.shape[1]
        # Pour chaque exemple : erreur = ce qu'on a prédit - ce qu'on voulait
        # Si A=0.8 et Y=1, alors dz=-0.2 (on a sous-estimé)
        # Si A=0.3 et Y=0, alors dz=+0.3 (on a surestimé)
        dz = A - Y

        # For each poids : Est-ce que ce poids me fait faire trop d'erreurs ?
        # Quand cette feature est forte, est-ce que je me trompe souvent ?
        # Le (1/m) fait la moyenne sur tous les exemples
        # Résult : un nombre pour chaque poids qui dit: ajuste-moi dans ce sens
        dw = (1/m) * np.dot(dz, X.T)

        # Bias : Est-ce que je suis globalement trop optimiste ou pessimiste ?
        # Je fais la moyenne de toutes mes erreurs
        # Si positif = je suis trop optimiste, si négatif = trop pessimiste
        db = (1/m) * np.sum(dz)

        # Corrige les poids : si dw dit "trop fort", je diminue (d'où le -)
        # alpha contrôle si : petite correction (0.01) ou grosse (0.5)
        self.__W = self.__W - alpha * dw

        # Correction du biais selon db
        # On ajuste le biais du neurone en fonction de la moyenne
        # des erreurs (db). Cela permet au neurone de s'adapter
        # globalement à la tendance des prédictions.
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Entraîne le neurone.

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
        # --- Validation des arguments ---
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
            A = self.forward_prop(X)
            cost = self.cost(Y, A)

            # --- si graph = True,
            # stocke tous les multiple de step + last iteration ---
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                iterations_list.append(i)

            # --- si verbose = True, alors print le coût si l'iteration
            # est un multiple de step ou la dernière iteration ---
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")

            self.gradient_descent(X, Y, A, alpha)

        # --- Si graph = True, affiche le graph
        if graph:
            plt.plot(iterations_list, costs, color='blue',linestyle='-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # --- Évaluation finale ---
        predictions, J = self.evaluate(X, Y)
        return predictions, float(J)
