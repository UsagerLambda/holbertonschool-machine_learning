#!/usr/bin/env python3
"""Update learning rate using inverse time in numpy."""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Met à jour alpha en utilisant la décroissance inverse du temps.

    Args :
        alpha (float) : taux d'apprentissage initial
        decay_rate (int) : poids utilisé pour déterminer la vitesse de
            décroissance de alpha
        global_step (int) : nombre de passages de descente de gradient
            effectués
        decay_step (int) : nombre de passages de descente de gradient
            avant que alpha ne décroit davantage
    """
    alpha = 1 / (1 + decay_rate * (global_step // decay_step)) * alpha
    return alpha
