#!/usr/bin/env python3
"""Train the Q-table for a Gym environnement."""

import numpy as np
epsilon_greedy = __import__("2-epsilon_greedy").epsilon_greedy


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """
    Train the Q-table for a Gym environnement.

    Using the epsilon-greedy strategy.
    """
    total_rewards = []
    for episode in range(episodes):
        episode_reward = 0

        state, _ = env.reset()
        step = 0
        done = False

        for step in range(max_steps):

            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, terminated, truncated, _ = env.step(action)
            if terminated and reward == 0:
                reward = -1
            done = terminated or truncated

            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state][action]
            )

            episode_reward += reward

            state = new_state

            if done:
                break

        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)

    return Q, total_rewards
