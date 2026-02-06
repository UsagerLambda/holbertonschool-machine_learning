#!/usr/bin/env python3
"""Module contenant la classe Transformer."""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformeur complet pour la traduction séquence-à-séquence."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1,
    ):
        """
        Initialise le transformeur.

        Args:
            N: nombre de blocs dans l'encodeur et le décodeur
            dm: dimensionnalité du modèle
            h: nombre de têtes d'attention
            hidden: nombre d'unités dans la couche cachée du FFN
            input_vocab: taille du vocabulaire d'entrée
            target_vocab: taille du vocabulaire cible
            max_seq_input: longueur maximale de la séquence d'entrée
            max_seq_target: longueur maximale de la séquence cible
            drop_rate: taux de dropout
        """
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask
    ):
        """
        Pass avant du transformeur.

        Args:
            inputs: tenseur d'entrée de forme (batch, input_seq_len)
            target: tenseur cible de forme (batch, target_seq_len)
            training: booléen indiquant si le modèle est en entraînement
            encoder_mask: masque de padding pour l'encodeur
            look_ahead_mask: masque pour l'auto-attention masquée du décodeur
            decoder_mask: masque de padding pour l'attention encodeur-décodeur

        Returns:
            Tenseur de sortie de forme (batch, target_seq_len, target_vocab)
        """
        encoder = self.encoder(inputs, training=training, mask=encoder_mask)
        decoder = self.decoder(
            target,
            encoder,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask,
        )
        output = self.linear(decoder)
        return output
