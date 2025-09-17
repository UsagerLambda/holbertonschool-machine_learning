#!/usr/bin/env python3
"""Module containing L2 regularization gradient descent implementation."""

import numpy as np
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculate le coût d'un réseau de neurones avec régularisation L2.

    Args:
        cost: tensor contenant le coût du réseau de neurones
            sans régularisation
        model: modèle Keras contenant les couches avec régularisation L2

    Returns:
        tensor contenant le coût total pour chaque couche du réseau,
        en tenant compte de la régularisation L2
    """
    costs = []
    for layer in model.layers:  # Pour chaque couche dans le réseau de neurones
        # Si layer a un attribut losses
        if hasattr(layer, 'losses') and layer.losses:
            # Somme des pénalités L2 dans losses
            layer_l2_loss = tf.add_n(layer.losses)
            # Add à la liste le coût original + la pénalité L2 de cette couche
            costs.append(cost + layer_l2_loss)

    # Renvoie un Tensor des coûts
    return tf.convert_to_tensor(costs, dtype=tf.float32)
