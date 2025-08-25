#!/usr/bin/env python3
import numpy as np

class Neuron:
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx  # Nombre de feature du neurone

        # Créer un tableau numpy de 1 ligne * nx colonnes
        # vecteur de poids aléatoires autour de 0
        # (valeurs positives et négatives, la plupart proches de 0)
        # En gros ça créer des valeurs de poids pour chaque features
        self.W = np.random.randn(1, nx)
        # Set le biais sur 0
        self.b = 0
        # Résultat de la sortie calculée par le neurone. (0 de base)
        self.A = 0
