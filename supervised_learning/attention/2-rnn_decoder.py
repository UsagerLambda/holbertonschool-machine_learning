#!/usr/bin/env python3
"""Module contenant le décodeur RNN avec attention."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention'"').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Décodeur RNN pour la traduction automatique."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialise le décodeur.

        Args:
            vocab: taille du vocabulaire de sortie
            embedding: dimension des embeddings
            units: nombre d'unités cachées du GRU
            batch: taille du batch
        """
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Donne forward du décodeur.

        Args:
            x: tensor (batch, 1) contenant le mot précédent
            s_prev: tensor (batch, units) état caché précédent
            hidden_states: tensor (batch, input_seq_len, units)
                sorties encodeur

        Returns:
            y: tensor (batch, vocab) prédiction du mot suivant
            hidden: tensor (batch, units) nouvel état caché
        """
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)

        x = tf.concat([context, x], axis=-1)
        x, s = self.gru(x)
        x = tf.reshape(x, (-1, x.shape[2]))

        y = self.F(x)

        return y, s
