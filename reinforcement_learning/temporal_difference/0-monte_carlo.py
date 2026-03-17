#!/usr/bin/env python3

import numpy as np
import gymnasium as gym


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))
            state = new_state
            if terminated or truncated:
                break
        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])
    return V
