#!/usr/bin/env python3
"""Gensim model to keras embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Gensim model to keras embedding layer."""
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors
    layer = tf.keras.layers.Embedding(
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            weights=[weights],
            trainable=False
        )
    return layer
