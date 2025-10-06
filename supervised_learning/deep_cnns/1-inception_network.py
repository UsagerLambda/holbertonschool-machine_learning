#!/usr/bin/env python3
"""Construit un réseau de neurones convolutifs basé sur l'arch Inception."""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Construit un réseau de neurones convolutifs basé sur l'arch Inception.

    L'architecture comprend plusieurs blocs d'inception,
    des couches de convolution, de pooling,
    de dropout et une couche de sortie dense avec une activation softmax.

    Returns:
        keras.Model: Le modèle Inception complet.
    """
    X = K.Input(shape=(224, 224, 3))

    A = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(
        2, 2), activation="relu", padding="same")(X)
    B = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding="same")(A)

    C = K.layers.Conv2D(filters=64, kernel_size=(
        1, 1), activation="relu", padding="same")(B)
    C1 = K.layers.Conv2D(filters=192, kernel_size=(
        3, 3), strides=(1, 1), activation="relu", padding="same")(C)
    D = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding="same")(C1)

    E = inception_block(D, [64, 96, 128, 16, 32, 32])  # inception (3a)
    F = inception_block(E, [128, 128, 192, 32, 96, 64])  # inception (3b)
    G = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding="same")(F)

    H = inception_block(G, [192, 96, 208, 16, 48, 64])  # inception (4a)
    Ie = inception_block(H, [160, 112, 224, 24, 64, 64])  # inception (4b)
    J = inception_block(Ie, [128, 128, 256, 24, 64, 64])  # inception (4c)
    K1 = inception_block(J, [112, 144, 288, 32, 64, 64])  # inception (4d)
    L = inception_block(K1, [256, 160, 320, 32, 128, 128])  # inception (4e)
    M = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding="same")(L)

    N = inception_block(M, [256, 160, 320, 32, 128, 128])  # inception (5a)
    Oe = inception_block(N, [384, 192, 384, 48, 128, 128])  # inception (5b)

    P = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(Oe)
    Q = K.layers.Dropout(rate=0.4)(P)
    R = K.layers.Dense(units=1000, activation="softmax")(Q)

    model = K.models.Model(inputs=X, outputs=R)
    return model
