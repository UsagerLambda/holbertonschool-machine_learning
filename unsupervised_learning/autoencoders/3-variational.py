#!/usr/bin/env python3
"""Fonction qui créer un autoencoder variationnel."""

import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """Layer de sampling avec KL divergence."""

    def call(self, inputs):
        """Layer de sampling avec KL divergence."""
        z_mean, z_log_var = inputs
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        # KL divergence
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var - keras.backend.square(
                z_mean) - keras.backend.exp(z_log_var), axis=1)
        kl_loss = keras.backend.mean(kl_loss)
        self.add_loss(kl_loss)

        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Créer un autoencoder variationnel.

    Args:
        input_dims (int): Entier contenant les dimensions de l'entrée du modèle
        hidden_layers (list): Liste contenant le nombre de nœuds pour chaque
            couche cachée de l'encodeur, respectivement. Les couches cachées
            sont inversées pour le décodeur.
        latent_dims (int): Entier contenant les dimensions de la représentation
            de l'espace latent

    Returns:
        tuple: (encoder, decoder, auto)
            encoder: Modèle encodeur
            decoder: Modèle décodeur
            auto: Modèle autoencoder complet
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for unit in hidden_layers:
        x = keras.layers.Dense(unit, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_input, [z, z_mean, z_log_var], name="encoder")

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for unit in reversed(hidden_layers):
        x = keras.layers.Dense(unit, activation='relu')(x)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_input, decoded, name="decoder")

    # VAE complet
    outputs = decoder(encoder(encoder_input)[0])
    auto = keras.Model(encoder_input, outputs, name="vae")

    # Loss function avec multiplication par input_dims
    def reconstruction_loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred) * input_dims

    auto.compile(optimizer='adam', loss=reconstruction_loss)

    return encoder, decoder, auto
