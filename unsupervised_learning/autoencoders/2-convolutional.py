#!/usr/bin/env python3
"""Fonction qui créer un autoencoder convolutionnel."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Fonction qui créer un autoencoder convolutionnel.

    Args:
        input_dims (tuple): tuple d'entiers contenant les dimensions de
            l'entrée du modèle
        filters (list): liste contenant le nombre de filtres pour chaque
            couche convolutionnelle de l'encodeur
        latent_dims (tuple): tuple d'entiers contenant les dimensions de
            la représentation de l'espace latent

    Returns:
        tuple: (encoder, decoder, auto)
            encoder (keras.Model): le modèle encodeur
            decoder (keras.Model): le modèle décodeur
            auto (keras.Model): le modèle autoencoder complet
    """
    input = keras.Input(shape=input_dims)

    x = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu', padding='same')(input)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    for dim in filters[1:]:
        x = keras.layers.Conv2D(
            dim, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs=input, outputs=x)

    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    all_filters = list(reversed(filters))

    for dim in all_filters[:-1]:
        x = keras.layers.Conv2D(
            dim, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        all_filters[-1], (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)

    decoder = keras.Model(inputs=latent_input, outputs=x)

    auto = keras.Model(inputs=input, outputs=decoder(encoder(input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
