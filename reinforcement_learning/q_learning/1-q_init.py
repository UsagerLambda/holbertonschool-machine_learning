#!/usr/bin/env python3
"""Initialize the Q-table of the environnement."""

import numpy as np


def q_init(env):
    """Initialize the Q-table of the environnement."""
    Qtable = np.zeros((env.observation_space.n, env.action_space.n))
    return Qtable
