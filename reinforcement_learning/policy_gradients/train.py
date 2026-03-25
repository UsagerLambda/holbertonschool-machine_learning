#!/usr/bin/env python3
"""Module d'entraînement par policy gradient (algorithme REINFORCE)."""

import gymnasium as gym
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Entraîne un agent avec l'algorithme REINFORCE (policy gradient).

    Args:
        env: environnement Gymnasium avec observation_space et action_space.
        nb_episodes (int): nombre d'épisodes d'entraînement.
        alpha (float): taux d'apprentissage. Défaut : 0.000045.
        gamma (float): facteur de discount pour les récompenses futures.
            Défaut : 0.98.
        show_result (bool): si True, affiche le rendu tous les 1000 épisodes.
            Défaut : False.

    Returns:
        list: liste des scores (récompenses cumulées) par épisode.
    """
    # init des poids
    n = env.observation_space.shape[0]
    m = env.action_space.n
    weight = np.random.rand(n, m)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()

        gradients = []
        rewards = []

        done = False
        score = 0

        # joue un épisode
        while not done:
            # Récupère une action
            action, grad = policy_gradient(state, weight)

            # Joue l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            # C'est la fin de l'épisode ?
            done = terminated or truncated

            # Save le gradient & les récompenses
            gradients.append(grad)
            rewards.append(reward)

            # Total des rewards de l'episode
            score += reward
            # Mise à jour de l'état
            state = next_state

        # calcul des returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        # normalisation (moyenne 0, std 1)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # update des poids
        for grad, G in zip(gradients, returns):
            weight += alpha * grad * G

        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

        # Rendu tous les 1000 épisodes si show_result est activé
        if show_result and episode % 1000 == 0:
            render_env = gym.make(env.spec.id, render_mode="human")
            state_r, _ = render_env.reset()
            done_r = False
            while not done_r:
                action_r, _ = policy_gradient(state_r, weight)
                state_r, _, term_r, trunc_r, _ = render_env.step(action_r)
                done_r = term_r or trunc_r
            render_env.close()

    return scores
