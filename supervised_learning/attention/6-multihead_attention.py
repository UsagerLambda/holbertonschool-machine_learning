#!/usr/bin/env python3
"""Multi Head Attention."""

import tensorflow as tf
sdp_attention = __import__("5-sdp_attention").sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Effectue l'attention multi-têtes."""

    def __init__(self, dm, h):
        """Initialise le layer avec dm (dimension) et h (nombre de têtes)."""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split(self, x, batch_size):
        """Découpe dm en (h, depth) et transpose."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        # Intervertie les dims 1 & 2
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Applique l'attention multi-têtes sur Q, K, V."""
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split(Q, batch_size)
        K = self.split(K, batch_size)
        V = self.split(V, batch_size)

        # Attention par tête
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)
        # repositionne les dims 1 & 2
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # de-split
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights
