#!/usr/bin/env python3
"""Display a game played by the trained DQN agent."""
import gymnasium as gym
from keras.models import load_model
from keras.optimizers.legacy import Adam
from rl.policy import GreedyQPolicy
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from train import FixWrappers


class FireOnLife(gym.Wrapper):
    """Press FIRE to launch the ball at the start of each life."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(1)
        self.lives = info.get("lives", 5) if isinstance(info, dict) else 5
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = info.get("lives", 0) if isinstance(info, dict) else 0
        if lives < self.lives and not done:
            obs, _, done, info = self.env.step(1)
        self.lives = lives
        return obs, reward, done, info

    def render(self, **kwargs):
        kwargs.pop('mode', None)
        return self.env.render(**kwargs)


def load_env():
    """Create the Breakout environment for playing."""
    from gymnasium.wrappers import AtariPreprocessing

    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="human")
    env = AtariPreprocessing(env)
    env = FixWrappers(env)
    env = FireOnLife(env)
    return env


def start():
    """Load the trained model and play."""
    model = load_model('best/best_policy.h5')
    env = load_env()

    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        policy=GreedyQPolicy(),
        memory=SequentialMemory(limit=10000, window_length=4),
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    dqn.test(env, nb_episodes=5, verbose=2)


if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt as e:
        print(f"\nInterrupted: {e}")
