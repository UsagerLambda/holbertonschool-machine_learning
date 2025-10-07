#!/usr/bin/env python3
"""Implémente un bloc de projection pour un réseau de neurones convolutif."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Implémente un bloc de projection pour un réseau de neurones convolutif.

    Args:
        A_prev (tf.Tensor): Tenseur d'entrée de la couche précédente, de forme
        filters (tuple): Tuple contenant trois entiers (F11, F3, F12),
            contenant le nombre de filtres pour chaque couche convolutive.
        s: stride of the first convolution

    Returns:
        tf.Tensor: La sortie du bloc d'identité, de même forme que A_prev.
    """
    he_init = K.initializers.HeNormal(seed=0)
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=he_init
    )(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    A = K.layers.Activation('relu')(norm1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=he_init
    )(A)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    B = K.layers.Activation('relu')(norm2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=he_init
    )(B)

    convbis = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        strides=(s, s),
        kernel_initializer=he_init
    )(A_prev)

    norm3 = K.layers.BatchNormalization(axis=3)(conv3)
    A_bis = K.layers.BatchNormalization(axis=3)(convbis)

    C = K.layers.Add()([norm3, A_bis])
    model = K.layers.Activation('relu')(C)

    return model
