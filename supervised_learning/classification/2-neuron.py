#!/usr/bin/env python3
"""Module qui définit la classe Neuron pour un réseau de neurones simple."""

import numpy as np


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
