#!/usr/bin/env python3
"""Construit une couche de transition pour un réseau DenseNet."""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Construit une couche de transition pour un réseau DenseNet.

    Args:
        X (tf.Tensor): La sortie de la couche précédente.
        nb_filters (int): Le nombre de filtres dans X.
        compression (float): Le facteur de compression pour
            réduire le nombre de filtres.

    Returns:
        tf.Tensor: La sortie de la couche de transition après le pooling.
        int: Le nouveau nombre de filtres après compression.
    """
    he_init = K.initializers.HeNormal(seed=0)
    norm = K.layers.BatchNormalization(axis=3)(X)
    new_nb_filters = int(compression * nb_filters)
    activ = K.layers.Activation('relu')(norm)
    conv = K.layers.Conv2D(
            filters=new_nb_filters,
            kernel_size=(1, 1),
            activation="linear",
            padding="same",
            kernel_initializer=he_init
        )(activ)
    avg_pool = K.layers.AveragePooling2D(pool_size=(
        2, 2), strides=(2, 2))(conv)

    return avg_pool, new_nb_filters
