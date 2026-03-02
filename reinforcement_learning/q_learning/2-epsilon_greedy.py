#!/usr/bin/env python3
"""Select the next action using the epsilon_greedy strategy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select the next action using the epsilon_greedy strategy."""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state])  # Récupère la meilleur action
    else:
        action = np.random.randint(Q.shape[1])  # Récupère une action random
    return action
