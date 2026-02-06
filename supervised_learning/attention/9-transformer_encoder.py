#!/usr/bin/env python3
"""Module contenant la classe Encoder pour les transformeurs."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encodeur complet d'un transformeur."""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_len,
            drop_rate=0.1
            ):
        """
        Initialise l'encodeur.

        Args:
            N: nombre de blocs d'encodeur
            dm: dimensionnalité du modèle
            h: nombre de têtes d'attention
            hidden: nombre d'unités dans la couche cachée du FFN
            input_vocab: taille du vocabulaire d'entrée
            max_seq_len: longueur maximale de séquence
            drop_rate: taux de dropout
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(
            dm,
            h,
            hidden,
            drop_rate
        ) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Pass avant de l'encodeur.

        Args:
            x: tenseur d'entrée de forme (batch, input_seq_len)
            training: booléen indiquant si le modèle est en entraînement
            mask: masque de padding à appliquer

        Returns:
            Tenseur de sortie de forme (batch, input_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        return x
