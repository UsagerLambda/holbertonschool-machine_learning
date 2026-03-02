#!/usr/bin/env python3
"""Create a FrozenLake gymnasium environnement."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Create a FrozenLake gymnasium environnement."""
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
        )
    return env
