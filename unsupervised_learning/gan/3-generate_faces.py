#!/usr/bin/env python3
"""Créer les méthodes d'un GAN."""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Méthode contenant deux méthodes.

    get_generator: créer un modèle de type générateur.
    get_discriminator: créer un modèle de type discriminateur.
    """

    def get_generator():
        """Créer le modèle générateur d'un GAN."""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation="tanh")(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)
        # ====================================================================
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=None,
            interpolation="nearest",
        )(x)
        x = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=None,
            interpolation="nearest",
        )(x)
        x = keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=None,
            interpolation="nearest",
        )(x)
        x = keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        generator = keras.Model(inputs, x, name="generator")
        return generator

    def get_discriminator():
        """Créer le modèle discriminateur d'un GAN."""
        inputs = keras.Input(shape=(16, 16, 1))
        # ====================================================================
        x = keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(inputs)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="valid"
        )(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        x = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="valid"
        )(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        x = keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="valid"
        )(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        x = keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="valid"
        )(x)
        x = keras.layers.Activation(keras.activations.tanh)(x)
        # ====================================================================
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1)(x)
        discriminator = keras.Model(inputs, x, name="discriminator")
        return discriminator

    return get_generator(), get_discriminator()
