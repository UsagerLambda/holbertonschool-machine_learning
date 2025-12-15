#!/usr/bin/env python3
"""Fonction qui créer un autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Fonction qui créer un autoencoder.

    Args:
        input_dims (int): un entier contenant
            les dimensions de l'entrée du modèle
        hidden_layers (list): une liste contenant le nombre de
            nœuds pour chaque couche cachée dans l'encodeur,
            respectivement les couches cachées doivent être inversées
            pour le décodeur
        latent_dims (int): un entier contenant les dimensions
            de la représentation de l'espace latent

    Returns:
        encoder: le modèle encodeur
        decoder: le modèle décodeur
        auto: le modèle complet de l'autoencoder
    """
    # Dimension des input pour l'encodeur
    input = keras.Input(shape=(input_dims,))

    # Dimension des layers entre l'entrée et le latent space
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(input)
    for unit in hidden_layers[1:]:
        encoded = keras.layers.Dense(unit, activation='relu')(encoded)
    # Dimension du latent space
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    # Sauvegarde le model d'encodage
    encoder = keras.Model(inputs=input, outputs=encoded)
    # Dimension des input pour le décodeur
    latent_input = keras.Input(shape=(latent_dims,))

    # Dimension des layers entre l'entrée et l'output (
    # le même que pour l'encodeur mais à l'envers)
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='sigmoid')(latent_input)
    for unit in reversed(hidden_layers[:-1]):
        decoded = keras.layers.Dense(unit, activation='sigmoid')(decoded)

    # Dimension de sortie (la même que celle d'entrée)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    # Sauvegarde le model de décodage
    decoder = keras.Model(inputs=latent_input, outputs=decoded)

    # Sauvegarde les deux modèle en un seul
    auto = keras.Model(inputs=input, outputs=decoder(encoder(input)))
    # Compile le modèle
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
