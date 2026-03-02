#!/usr/bin/env python3
"""Let the agent play an episode."""

import numpy as np


def play(env, Q, max_steps=100):
    """Let the agent play an episode."""
    state, _ = env.reset()
    done = False
    total_reward = 0
    rendered_states = []
    for step in range(max_steps):
        rendered_states.append(env.render())  # Enr. l'état de la grille
        action = np.argmax(Q[state])  # Récupère le meilleur mouvement

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = new_state

        if done:
            break

    rendered_states.append(env.render())  # Enr. l'état final de la grille
    return total_reward, rendered_states
