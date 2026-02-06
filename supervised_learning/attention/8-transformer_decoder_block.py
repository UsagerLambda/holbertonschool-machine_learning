#!/usr/bin/env python3
"""Module contenant la classe DecoderBlock pour les transformeurs."""

import tensorflow as tf

MultiHeadAttention = __import__("6-multihead_attention").MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Bloc de décodeur pour un transformeur."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialise le bloc de décodeur.

        Args:
            dm: dimensionnalité du modèle
            h: nombre de têtes d'attention
            hidden: nombre d'unités dans la couche cachée du FFN
            drop_rate: taux de dropout
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Pass avant du bloc de décodeur.

        Args:
            x: tenseur d'entrée de forme (batch, target_seq_len, dm)
            encoder_output: sortie de l'encodeur (batch, input_seq_len, dm)
            training: booléen indiquant si le modèle est en entraînement
            look_ahead_mask: masque pour l'auto-attention masquée
            padding_mask: masque de padding pour l'attention encodeur-décodeur

        Returns:
            Tenseur de sortie de forme (batch, target_seq_len, dm)
        """
        output, _ = self.mha1(x, x, x, look_ahead_mask)
        drop1 = self.dropout1(output, training=training)
        norm1 = self.layernorm1(x + drop1)

        output2, _ = self.mha2(
            norm1, encoder_output, encoder_output, padding_mask)
        drop2 = self.dropout2(output2, training=training)
        norm2 = self.layernorm2(norm1 + drop2)

        feed_forward = self.dense_hidden(norm2)
        feed_forward = self.dense_output(feed_forward)
        drop3 = self.dropout3(feed_forward, training=training)
        norm3 = self.layernorm3(norm2 + drop3)

        return norm3
