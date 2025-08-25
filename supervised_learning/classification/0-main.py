#!/usr/bin/env python3

import numpy as np

Neuron = __import__('0-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)  # graine pour avoir toujours les même randn
neuron = Neuron(X.shape[0])  # Crée un objet Neuron avec nx = X.shape[0]
print(neuron.W)  # Affiche le vecteur de poids (tableau 1 ligne x nx colonnes)
print(neuron.W.shape)  # Affiche la forme du tableau de poids 1 x nx
print(neuron.b)  # Affiche valeur de b (initialisé à 0)
print(neuron.A)  # Affiche valeur de A (initialisé à 0)
neuron.A = 10  # modifie la valeur de l'attribut public A par 10
print(neuron.A)  # Affiche valeur de A (maintenant 10)
