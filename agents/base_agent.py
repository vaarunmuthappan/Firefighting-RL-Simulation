"""Abstract base class for all RL and heuristic agents."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np


class BaseAgent(ABC):
    """Abstract base class defining the agent interface.

    All concrete agents (DQN, PPO, Heuristic, …) must inherit from this
    class and implement every abstract method listed below. This ensures
    a uniform API for training, evaluation, and checkpointing.

    Attributes:
        env: The gymnasium environment the agent interacts with.
        config: Agent-level configuration dictionary.
    """

    def __init__(self, env, config: dict) -> None:
        """Initialise the base agent.

        Args:
            env: A gymnasium-compatible environment instance.
            config: Dictionary of agent hyperparameters and settings.
        """
        self.env = env
        self.config = config

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> int:
        """Select an action given the current observation.

        Args:
            obs: Observation array from the environment.

        Returns:
            Integer action index.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, total_timesteps: int, callbacks: Optional[List[Any]] = None) -> None:
        """Run the training loop for the specified number of timesteps.

        Args:
            total_timesteps: Total environment steps to train for.
            callbacks: Optional list of SB3-compatible callback objects.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent's model weights to disk.

        Args:
            path: File path (without extension) where the model is saved.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: File path (with or without extension) of the saved model.
        """
        raise NotImplementedError
