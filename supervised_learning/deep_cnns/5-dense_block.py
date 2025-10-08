#!/usr/bin/env python3
"""Construit un bloc dense pour un réseau DenseNet."""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Construit un bloc dense pour un réseau DenseNet.

    Args:
        X (tf.Tensor): La sortie de la couche précédente.
        nb_filters (int): Le nombre de filtres dans X.
        growth_rate (int): Le taux de croissance pour le bloc dense.
        layers (int): Le nombre de couches dans le bloc dense.

    Returns:
        tf.Tensor: La sortie du bloc dense.
        int: Le nombre total de filtres après le bloc dense.
    """
    he_init = K.initializers.HeNormal(seed=0)

    # Boucle pour créer le nombre spécifié de couches dans le bloc dense
    for _ in range(layers):
        dense = K.layers.BatchNormalization(axis=3)(X)
        dense = K.layers.Activation('relu')(dense)
        dense = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            activation="linear",
            padding="same",
            kernel_initializer=he_init
        )(dense)
        dense = K.layers.BatchNormalization(axis=3)(dense)
        dense = K.layers.Activation('relu')(dense)
        dense = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            activation="linear",
            padding="same",
            kernel_initializer=he_init
        )(dense)

        # Mise à jour de X en concaténant la sortie de la couche
        # actuelle avec l'entrée existante
        # Cela permet de conserver les caractéristiques des couches précédentes
        X = K.layers.Concatenate()([X, dense])
        # Mise à jour du nombre total de filtres après l'ajout
        # de la nouvelle couche
        nb_filters += growth_rate

    # Retourne la sortie du bloc dense et le nombre total de filtres
    return X, nb_filters
