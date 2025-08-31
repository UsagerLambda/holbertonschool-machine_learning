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
    model = K.Sequential()  # Créer un modèle séquentiel

    for i in range(len(layers)):  # Boucle pour chaque couches
        if i == 0:  # Première couche
            model.add(K.layers.Dense(
                layers[i],  # Nombre de neurones
                activation=activations[i],  # Type d'activation
                # Régularisation L2 sur les poids de la couche
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,),  # Forme d'entré
                name=f"dense"  # Nom de la couche
            ))
        else:  # Couches suivantes
            model.add(K.layers.Dense(
                layers[i],  # Nombre de neurones
                activation=activations[i],  # Type d'activation
                # Régularisation L2 sur les poids de la couche
                kernel_regularizer=K.regularizers.l2(lambtha),
                name=f"dense_{i}"  # Nom de la couche
            ))
        if i != len(layers) - 1:  # Si pas la dernière couche
            # Désactive aléatoirement une fraction des neurones de la couche
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
