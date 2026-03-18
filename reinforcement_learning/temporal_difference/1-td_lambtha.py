#!/usr/bin/env python3
"""Module implémentant l'algorithme TD(λ) pour estimer la fonction valeur."""

import numpy as np


def td_lambtha(
    env,
    V,
    policy,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
):
    """Effectue l'algorithme TD(λ) pour estimer la fonction valeur.

    Args:
        env: Instance d'un environnement compatible OpenAI Gym.
        V (numpy.ndarray): Estimations de valeur de forme (s,) par état.
        policy (callable): Fonction qui associe un état à une action.
        lambtha (float): Facteur de décroissance des traces d'éligibilité (λ).
        episodes (int): Nombre d'épisodes à exécuter. Par défaut 5000.
        max_steps (int): Nombre maximal de pas par épisode. Par défaut 100.
        alpha (float): Taux d'apprentissage. Par défaut 0.1.
        gamma (float): Facteur de réduction. Par défaut 0.99.

    Returns:
        numpy.ndarray: Estimations de valeur V mises à jour, de forme (s,).
    """
    for episode in range(episodes):
        state, _ = env.reset()
        e = np.zeros(V.shape)

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[new_state] - V[state]
            e = gamma * lambtha * e
            e[state] += 1
            V = V + alpha * delta * e
            if terminated or truncated:
                break
            state = new_state

    return V
