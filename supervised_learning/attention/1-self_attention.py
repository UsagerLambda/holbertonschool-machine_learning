#!/usr/bin/env python3
"""Module de Self-Attention pour la traduction automatique."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calcule l'attention pour la traduction (Bahdanau)."""

    def __init__(self, units):
        """Initialise la couche avec units unités cachées."""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Retourne le vecteur de contexte et les poids d'attention."""
        # hidden_states = (32, 10, 256)
        s_prev_ext = tf.expand_dims(s_prev, 1)  # Deviens (32, 1, 256)

        # Transformation en layers
        W_s = self.W(s_prev_ext)
        U_h = self.U(hidden_states)

        # Score d'attention
        score = self.V(tf.nn.tanh(W_s + U_h))

        # Normalisation
        weights = tf.nn.softmax(score, axis=1)

        # Somme pondérée des mots sources
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
