"""Evaluation utilities for trained agents.

Provides ``evaluate_agent``, a simple function that runs N episodes with a
given agent and returns summary statistics. Used both during training (via
EvalCallback) and for post-training analysis.
"""
from typing import Dict, List

import numpy as np


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 5,
    render: bool = False,
) -> Dict[str, object]:
    """Run ``num_episodes`` evaluation episodes and return summary statistics.

    The agent's ``get_action`` method is used to select actions deterministically.
    If ``agent`` is an SB3 model (DQNAgent / PPOAgent), its underlying
    ``model.predict`` is called via the agent wrapper.

    Args:
        agent: An instance of BaseAgent (DQNAgent, PPOAgent, or HeuristicAgent),
            or any object with a ``get_action(obs) -> int`` method.
        env: A gymnasium-compatible environment instance.
        num_episodes: Number of episodes to evaluate (default 5).
        render: If True, call ``env.render()`` at the start of each episode to
            enable visual playback (requires a display).

    Returns:
        Dict with keys:
            - ``"mean_reward"`` (float): Mean episode reward.
            - ``"std_reward"`` (float): Standard deviation of episode rewards.
            - ``"mean_length"`` (float): Mean episode length in steps.
            - ``"episode_rewards"`` (list[float]): Per-episode total rewards.
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(num_episodes):
        obs, _ = env.reset()

        if render:
            env.render()

        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_length += 1
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "episode_rewards": episode_rewards,
    }
