#!/usr/bin/env python3
"""Module contenant la classe Decoder pour les transformeurs."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Décodeur complet d'un transformeur."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        target_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """
        Initialise le décodeur.

        Args:
            N: nombre de blocs de décodeur
            dm: dimensionnalité du modèle
            h: nombre de têtes d'attention
            hidden: nombre d'unités dans la couche cachée du FFN
            target_vocab: taille du vocabulaire cible
            max_seq_len: longueur maximale de séquence
            drop_rate: taux de dropout
        """
        super().__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training,
        look_ahead_mask,
        padding_mask
    ):
        """
        Pass avant du décodeur.

        Args:
            x: tenseur d'entrée de forme (batch, target_seq_len)
            encoder_output: sortie de l'encodeur
                (batch, input_seq_len, dm)
            training: booléen indiquant si le modèle est en entraînement
            look_ahead_mask: masque pour l'auto-attention masquée
            padding_mask: masque de padding pour l'attention
                encodeur-décodeur

        Returns:
            Tenseur de sortie de forme (batch, target_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(
                x,
                encoder_output=encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )
        return x
