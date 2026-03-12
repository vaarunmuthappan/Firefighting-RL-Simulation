"""PPO agent wrapping stable-baselines3 PPO."""
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents.base_agent import BaseAgent
from agents.dqn_agent import ChannelLastCNN


class PPOAgent(BaseAgent):
    """PPO agent backed by stable-baselines3.

    Uses CnnPolicy with a custom channel-last CNN for image observations or
    MlpPolicy for flat vector observations, depending on the ``policy`` config.

    Shares the same ``ChannelLastCNN`` features extractor as DQNAgent so both
    agents have a consistent architecture.

    Attributes:
        model: The underlying SB3 PPO model instance.
    """

    def __init__(self, env, config: dict) -> None:
        """Initialise PPOAgent and build the SB3 PPO model.

        Args:
            env: A gymnasium-compatible environment.
            config: Dict with optional keys:
                - policy (str): "CnnPolicy" or "MlpPolicy" (default "CnnPolicy")
                - learning_rate (float): default 3e-4
                - n_steps (int): default 2048
                - batch_size (int): default 64
                - n_epochs (int): default 10
                - gamma (float): default 0.99
                - gae_lambda (float): default 0.95
                - clip_range (float): default 0.2
                - verbose (int): default 1
        """
        super().__init__(env, config)

        policy = config.get("policy", "CnnPolicy")

        # For CnnPolicy: inject the custom channel-last CNN features extractor
        # and disable SB3's internal image normalisation (obs is already [0, 1]).
        policy_kwargs = dict(config.get("policy_kwargs", {}))
        if policy == "CnnPolicy":
            policy_kwargs.setdefault("features_extractor_class", ChannelLastCNN)
            policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
            policy_kwargs.setdefault("normalize_images", False)

        self.model = PPO(
            policy,
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            verbose=config.get("verbose", 1),
            policy_kwargs=policy_kwargs,
        )

    def get_action(self, obs: np.ndarray) -> int:
        """Select a greedy action given an observation.

        Args:
            obs: Observation array (H, W, C) float32 from the environment.

        Returns:
            Integer action index (deterministic inference).
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def train(
        self,
        total_timesteps: int,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        """Train the PPO model.

        Args:
            total_timesteps: Total environment interaction steps.
            callbacks: Optional list of SB3 BaseCallback instances.
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

    def save(self, path: str) -> None:
        """Save model weights to ``path``.

        Args:
            path: Destination file path (SB3 appends ".zip" automatically).
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model weights from ``path``.

        Args:
            path: Source file path of a previously saved PPO model.
        """
        self.model = PPO.load(path, env=self.env)
