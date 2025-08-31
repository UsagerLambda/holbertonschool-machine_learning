#!/usr/bin/env python3
"""Initialize a neural network using Keras."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network using Keras library.

    Args:
        nx (int): number of input features to the network
        layers (list): list containing the number of nodes
            in each layer of the network
        activations (list): list containing the activation functions
            used for each layer of the network
        lambtha (): L2 regularization parameter
        keep_prob (): the probability that node will be kept for dropout
    """
    # Entrée du réseau avec nx features en entrée
    inputs = K.Input(shape=(nx,))
    layer = inputs
    for i in range(len(layers)):  # Boucle dans les couches cachées
        layer = K.layers.Dense(  # Créer la couche de neurones
            layers[i],  # Nombre de neurones
            activations[i],  # Fonction d'activation
            kernel_regularizer=K.regularizers.l2(lambtha)  # Pénalitée L2
        )(layer)

        if keep_prob < 1:  # Désactive des neurones aléatoirement
            layer = K.layers.Dropout(1 - keep_prob)(layer)

    outputs = K.layers.Dense(  # Pour la dernière couche
        layers[-1],  # Nombre de neurones
        activation=activations[-1],  # Fonction d'activation
        kernel_regularizer=K.regularizers.l2(lambtha)  # Pénalitée L2
    )(layer)

    # Relie les couches cachées et la couche de sortie
    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
