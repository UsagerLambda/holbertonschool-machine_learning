#!/usr/bin/env python3
"""Module implémentant un encodeur RNN pour la traduction automatique."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodeur RNN utilisant une couche GRU pour encoder une séquence."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialise l'encodeur RNN.

        Args:
            vocab: Taille du vocabulaire d'entrée.
            embedding: Dimension des vecteurs d'embedding.
            units: Nombre d'unités cachées dans la couche GRU.
            batch: Taille du batch.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            kernel_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True,
        )

    def initialize_hidden_state(self):
        """Initialise la "mémoire" du GRU."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Effectue la passe avant de l'encodeur."""
        # Transforme les mots en vecteurs d'embedding
        embedded = self.embedding(x)
        # Pour chaque mot le GRU va créer un contexte qui va s'étoffer
        # au fur et à mesure qu'il parcours les mots de la phrase
        # le hidden représente une version résumé de la phrase après avoir
        # tous parcourus
        outputs, hidden = self.gru(embedded, initial_state=initial)
        return outputs, hidden
