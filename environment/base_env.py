"""Abstract base class for firefighting RL environments.

All concrete environments must inherit from BaseFireEnv and implement the
abstract methods defined here. This class enforces the gymnasium interface
contract and provides shared state initialisation.

Gymnasium interface contract
----------------------------
- reset(seed, options) -> (obs, info)
    Called at the start of every episode. Must return the initial observation
    and an (optionally empty) info dict.
- step(action) -> (obs, reward, terminated, truncated, info)
    Called at each environment timestep. Returns the next observation, the
    scalar reward, two booleans indicating episode end (terminated = natural
    end, truncated = time-limit cut-off), and an info dict.
- render() -> None
    Activates visual rendering of the environment.
- _build_observation() -> np.ndarray
    Internal method that assembles and normalises the observation array.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec


class BaseFireEnv(gym.Env, ABC):
    """Abstract base class for all firefighting RL environments.

    Subclasses must implement the abstract methods below. Common state such
    as ``agent_pos``, ``num_steps``, and the gymnasium ``spec`` are
    initialised here so that concrete classes can focus on domain logic.

    Attributes:
        agent_pos: Current [row, col] position of the agent on the map.
        num_steps: Number of environment steps taken in the current episode.
        spec: Gymnasium EnvSpec used by wrappers and vectorised environments.
    """

    def __init__(self) -> None:
        """Initialise shared environment state."""
        super().__init__()
        self.agent_pos: list = [0, 0]
        self.num_steps: int = 0
        self.spec: Optional[EnvSpec] = None

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state.

        Subclasses must call ``super().reset(seed=seed)`` to seed the internal
        gymnasium RNG (``self.np_random``) before doing their own reset logic.

        Args:
            seed: Optional RNG seed for reproducible episodes.
            options: Optional dict with extra reset parameters (ignored by
                most implementations).

        Returns:
            obs: Initial observation array of shape (H, W, C).
            info: Auxiliary information dict (may be empty).
        """
        # Delegate to gym.Env to seed self.np_random; do NOT raise here.
        super().reset(seed=seed)

    @abstractmethod
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run one environment timestep.

        Args:
            action: Integer action selected by the agent.

        Returns:
            obs: Observation after taking the action, shape (H, W, C).
            reward: Scalar reward signal.
            terminated: True when the episode ends naturally (fire out).
            truncated: True when the episode is cut off by a step limit.
            info: Auxiliary information dict.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        """Activate visual rendering of the environment."""
        raise NotImplementedError

    @abstractmethod
    def _build_observation(self) -> np.ndarray:
        """Assemble and normalise the multi-channel observation array.

        Returns:
            Float32 array of shape (screen_size, screen_size, num_channels).
        """
        raise NotImplementedError
