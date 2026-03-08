#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout."""
import numpy as np
import gymnasium as gym
from tensorflow.keras.layers import Dense, Flatten, Permute, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.agents.dqn import DQNAgent
from rl.callbacks import Callback


class FixWrappers(gym.Wrapper):
    """Make Gymnasium compatible with keras-rl."""

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return np.array(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return (obs, reward, done, info)

    def render(self, **kwargs):
        kwargs.pop('mode', None)
        return self.env.render(**kwargs)


def load_env(render_mode=None):
    """Create and wrap the Breakout environment."""
    from gymnasium.wrappers import AtariPreprocessing

    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env)
    env = FixWrappers(env)
    return env


def init_model(env):
    """Build the CNN model (Nature DQN architecture)."""
    n_action = env.action_space.n
    INPUT_SHAPE = (4,) + env.observation_space.shape

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=INPUT_SHAPE))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_action, activation='linear'))

    return model, n_action


class SaveBestModel(Callback):
    """Save the model whenever the mean episode reward improves."""

    def __init__(self, keras_model, filepath='policy_best.h5'):
        self.keras_model = keras_model
        self.filepath = filepath
        self.best_reward = -np.inf
        self.episode_rewards = []

    def on_episode_end(self, episode, logs):
        reward = logs.get('episode_reward', 0)
        self.episode_rewards.append(reward)
        mean_reward = np.mean(self.episode_rewards[-100:])
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.keras_model.save(self.filepath)
            print(f"\nMeilleur modèle sauvegardé (mean reward: {mean_reward:.2f})")


def train(env, model, n_action, nb_steps=1500000):
    """Train the DQN agent."""
    memory = SequentialMemory(limit=1000000, window_length=4)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=nb_steps // 4
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50000,
        train_interval=4,
        gamma=0.99,
        target_model_update=10000,
        delta_clip=1.0,
        enable_double_dqn=True
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    dqn.fit(env, nb_steps=nb_steps, callbacks=[SaveBestModel(model)])
    model.save('policy.h5')
    env.close()


def start():
    """Entry point."""
    env = load_env()
    model, n_action = init_model(env)
    train(env, model, n_action)


if __name__ == '__main__':
    start()
