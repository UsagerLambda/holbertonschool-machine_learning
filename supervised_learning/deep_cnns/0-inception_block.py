#!/usr/bin/env python3
"""Construit un bloc d'inception."""


from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Construit un bloc d'inception.

    Args:
        A_prev (tensor): la sortie de la couche précédente.
        filters (tuple ou list): contient respectivement :
            - F1 : nb de filtres dans la conv 1x1.
            - F3R : nb de filtres dans la conv 1x1 avant la convolution 3x3.
            - F3 : nb de filtres dans la conv 3x3.
            - F5R : nb de filtres dans la conv 1x1 avant la convolution 5x5.
            - F5 : nb de filtres dans la conv 5x5.
            - FPP : nb de filtres dans la conv 1x1 après le max pooling.

    Toutes les convolutions à l'intérieur du bloc d'inception utilisent
        une activation ReLU (Rectified Linear Unit).

    Returns:
        tensor: la sortie concaténée du bloc d'inception.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # F1 : convolution 1x1
    conv_F1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        activation="relu",
        padding="same",
    )(A_prev)

    # F3R : convolution 1x1 avant la convolution 3x3.
    conv_F3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        activation="relu",
        padding="valid",
    )(A_prev)

    # F3 : convolution 3x3
    conv_F3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    )(conv_F3R)

    # F5R : convolution 1x1 avant la convolution 5x5.
    conv_F5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        activation="relu",
        padding="valid",
    )(A_prev)

    # F5 : convolution 5x5
    conv_F5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        activation="relu",
        padding="same",
    )(conv_F5R)

    # Pooling
    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same"
    )(A_prev)

    # FPP : convolution 1x1
    conv_FPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        activation="relu",
        padding="same",
    )(max_pool)

    model = K.layers.Concatenate(axis=3)([conv_F1, conv_F3, conv_F5, conv_FPP])
    return model
