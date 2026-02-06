#!/usr/bin/env python3
"""Module contenant la classe EncoderBlock pour les transformeurs."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Bloc d'encodeur pour un transformeur."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialise le bloc d'encodeur.

        Args:
            dm: dimensionnalité du modèle
            h: nombre de têtes d'attention
            hidden: nombre d'unités dans la couche cachée du FFN
            drop_rate: taux de dropout
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Pass avant du bloc d'encodeur.

        Args:
            x: tenseur d'entrée de forme (batch, seq_len, dm)
            training: booléen indiquant si le modèle est en entraînement
            mask: masque optionnel à appliquer

        Returns:
            Tenseur de sortie de forme (batch, seq_len, dm)
        """
        output, _ = self.mha(x, x, x, mask)
        drop1 = self.dropout1(output, training=training)
        norm1 = self.layernorm1(x + drop1)

        feed_forward = self.dense_hidden(norm1)
        feed_forward = self.dense_output(feed_forward)
        drop2 = self.dropout2(feed_forward, training=training)
        norm2 = self.layernorm2(norm1 + drop2)

        return norm2
