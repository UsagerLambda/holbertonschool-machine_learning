#!/usr/bin/env python3
"""Scaled Dot Product Attention."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention.

    Args:
        Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
        K: tensor with shape (..., seq_len_v, dk) containing the key matrix
        V: tensor with shape (..., seq_len_v, dv) containing the value matrix
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
              containing the optional mask, or defaulted to None

    Returns:
        output: tensor with shape (..., seq_len_q, dv) containing the
                scaled dot product attention
        weights: tensor with shape (..., seq_len_q, seq_len_v) containing
                 the attention weights
    """
    matmul = tf.matmul(Q, K, transpose_b=True)

    # Normalisation
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = matmul / tf.math.sqrt(dk)

    # Ajoute -∞ aux position à ignorer
    if mask is not None:
        scaled += mask * -1e9

    # Convertie en probabilités
    attention_w = tf.nn.softmax(scaled, axis=-1)
    # Moyenne pondérée
    output = tf.matmul(attention_w, V)

    return output, attention_w
