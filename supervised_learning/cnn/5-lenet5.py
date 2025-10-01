#!/usr/bin/env python3
"""Construit l'architecture LeNet-5 pour la classification d'images."""


from tensorflow import keras as K


def lenet5(X):
    """
    Construit l'architecture LeNet-5 pour la classification d'images.

    LeNet-5 est un réseau de neurones convolutifs classique conçu pour la
    reconnaissance d'images, composé de couches convolutives, de couches de
    pooling et de couches entièrement connectées. Cette implémentation utilise
    l'initialisation he_normal (graine 0) pour toutes les couches nécessitant
    une initialisation des poids, et la fonction d'activation ReLU pour les
    couches cachées.

    Paramètres :
        X : K.Input de forme (m, 28, 28, 1)
            Tensor contenant les images d'entrée pour le réseau, où m est le
            nombre d'images.

    Architecture du modèle :
        - Couche convolutionnelle avec 6 noyaux de taille 5x5, padding 'same',
          activation ReLU
        - Couche de max pooling avec noyaux 2x2 et strides 2x2
        - Couche convolutionnelle avec 16 noyaux de taille 5x5, padding
          'valid', activation ReLU
        - Couche de max pooling avec noyaux 2x2 et strides 2x2
        - Couche entièrement connectée avec 120 neurones, activation ReLU
        - Couche entièrement connectée avec 84 neurones, activation ReLU
        - Couche de sortie entièrement connectée avec 10 neurones, activation
          softmax

    Toutes les couches nécessitant une initialisation utilisent la méthode
    he_normal avec la graine 0 pour garantir la reproductibilité.

    Retourne :
        Un objet K.Model compilé avec l'optimiseur Adam (hyperparamètres par
        défaut), la fonction de perte categorical_crossentropy et la métrique
        accuracy.
    """
    """"""
    # Merci Yann LeCun
    # Initialisateur he_normal avec la graine 0
    he_init = K.initializers.HeNormal(seed=0)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding,
    # he_normal initializer, relu activation
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        activation='relu',
        padding='same',
        kernel_initializer=he_init
    )(X)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding,
    # he_normal initializer, relu activation
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        activation='relu',
        padding='valid',
        kernel_initializer=he_init
    )(pool1)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Aplatie le sortie
    flat = K.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes, he_normal initializer,
    # relu activation
    dense1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=he_init
    )(flat)
    # Fully connected layer with 84 nodes, he_normal initializer,
    # relu activation
    dense2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=he_init
    )(dense1)
    # Fully connected softmax output layer with 10 nodes,
    # he_normal initializer
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=he_init
    )(dense2)

    # Création du modèle
    model = K.Model(inputs=X, outputs=output)

    # K.Model compiled to use Adam optimization (with default hyperparameters)
    # and accuracy metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
