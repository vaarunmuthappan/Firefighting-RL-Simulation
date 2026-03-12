"""DQN agent wrapping stable-baselines3 DQN."""
from typing import Any, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents.base_agent import BaseAgent


class ChannelLastCNN(BaseFeaturesExtractor):
    """Custom CNN features extractor that accepts channel-last (H, W, C) float32 input.

    This replaces SB3's NatureCNN which expects channel-first uint8 images.
    The network transposes (H, W, C) → (C, H, W) internally and applies a
    small convolutional stack followed by a linear head.

    Args:
        observation_space: The (H, W, C) float32 Box observation space.
        features_dim: Size of the output feature vector (default 256).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        # channel-last: (H, W, C) → n_channels = shape[-1]
        h, w, n_channels = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, h, w)
            flat_size = int(self.cnn(dummy).shape[1])
        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input: (batch, H, W, C) → transpose to (batch, C, H, W)
        if observations.ndim == 4:
            x = observations.permute(0, 3, 1, 2)
        else:
            # Single obs (H, W, C) → (1, C, H, W)
            x = observations.permute(2, 0, 1).unsqueeze(0)
        return self.linear(self.cnn(x))


class DQNAgent(BaseAgent):
    """DQN agent backed by stable-baselines3.

    Uses CnnPolicy with a custom channel-last CNN for image observations or
    MlpPolicy for flat vector observations, depending on the ``policy`` config.

    The custom ``ChannelLastCNN`` features extractor accepts the environment's
    native (H, W, C) float32 observations directly without requiring any
    channel transposition or uint8 conversion.

    Attributes:
        model: The underlying SB3 DQN model instance.
    """

    def __init__(self, env, config: dict) -> None:
        """Initialise DQNAgent and build the SB3 DQN model.

        Args:
            env: A gymnasium-compatible environment.
            config: Dict with optional keys:
                - policy (str): "CnnPolicy" or "MlpPolicy" (default "CnnPolicy")
                - learning_rate (float): default 1e-4
                - buffer_size (int): default 50000
                - batch_size (int): default 32
                - gamma (float): default 0.99
                - train_freq (int): default 4
                - target_update_interval (int): default 1000
                - exploration_fraction (float): default 0.3
                - exploration_final_eps (float): default 0.05
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

        self.model = DQN(
            policy,
            env,
            learning_rate=config.get("learning_rate", 1e-4),
            buffer_size=config.get("buffer_size", 50000),
            batch_size=config.get("batch_size", 32),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 4),
            target_update_interval=config.get("target_update_interval", 1000),
            exploration_fraction=config.get("exploration_fraction", 0.3),
            exploration_final_eps=config.get("exploration_final_eps", 0.05),
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
        """Train the DQN model.

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
            path: Source file path of a previously saved DQN model.
        """
        self.model = DQN.load(path, env=self.env)
