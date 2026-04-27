"""PPO agent wrapping stable-baselines3 PPO.

PPO is preferred over DQN when sample efficiency (fewest episodes to learn)
matters because:
  - No replay-buffer warm-up: learning starts from step 1 (DQN wastes the
    first `learning_starts` steps doing random actions before any gradient
    update).
  - On-policy updates avoid stale-data bias in the multi-agent sequential
    setting where the policy changes rapidly.
  - The entropy bonus (ent_coef) provides built-in exploration without
    needing an explicit ε-greedy schedule.

Architecture: same ChannelLastCNN as DQN — accepts (H, W, C) float32 obs.
"""
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents.base_agent import BaseAgent


class ChannelLastCNN(BaseFeaturesExtractor):
    """Custom CNN features extractor for channel-last (H, W, C) float32 obs.

    Identical architecture to the DQN version so weights are compatible if
    you ever want to warm-start one from the other.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
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
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, h, w)
            flat_size = int(self.cnn(dummy).shape[1])
        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 4:
            x = observations.permute(0, 3, 1, 2)
        else:
            x = observations.permute(2, 0, 1).unsqueeze(0)
        return self.linear(self.cnn(x))


class PPOAgent(BaseAgent):
    """PPO agent backed by stable-baselines3.

    Recommended hyperparameters for fast learning with fewest episodes:
      - n_steps=512: collect a short rollout before each update (low latency)
      - n_epochs=10: reuse each rollout 10× for better data efficiency
      - ent_coef=0.01: small entropy bonus keeps exploration alive without
        an explicit ε schedule
      - learning_rate=3e-4: PPO standard; higher than DQN's 1e-4 is safe
        due to the clipped objective

    Attributes:
        model: The underlying SB3 PPO model instance.
    """

    def __init__(self, env, config: dict) -> None:
        super().__init__(env, config)

        policy = config.get("policy", "CnnPolicy")

        policy_kwargs = dict(config.get("policy_kwargs", {}))
        if policy == "CnnPolicy":
            policy_kwargs.setdefault("features_extractor_class", ChannelLastCNN)
            policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
            policy_kwargs.setdefault("normalize_images", False)

        self.model = PPO(
            policy,
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 512),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.01),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            verbose=config.get("verbose", 1),
            policy_kwargs=policy_kwargs,
        )

    def get_action(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def train(self, total_timesteps: int, callbacks: Optional[List[Any]] = None) -> None:
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.env)
