"""Rule-based heuristic agent for the firefighting RL environment.

The agent moves toward the nearest burning cell and places a fireline at
every cell it visits that is currently unburned. This provides a simple
non-learning baseline to compare against trained RL agents.
"""
from typing import Any, List, Optional

import numpy as np

from agents.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """Rule-based agent that chases and contains the nearest fire.

    Strategy
    --------
    1. Find all cells whose BurnStatus value == 1 (BURNING) in the fire map.
    2. Compute Manhattan distance from the agent's current position to each
       burning cell.
    3. Move one step toward the nearest burning cell (up/down/left/right).
    4. Always select the fireline interaction (interaction index 1 in the
       discrete action space).

    The action is encoded following the same convention as FireEnv::

        action = interaction_idx * len(movements) + movement_idx

    Movements order: ["none", "up", "down", "left", "right"] (indices 0–4).
    Interactions order: ["none", "fireline"] (indices 0–1).
    Action space size: 5 * 2 = 10.
    """

    # Movement index constants (matching FireEnv conventions)
    _MOVE_NONE = 0
    _MOVE_UP = 1
    _MOVE_DOWN = 2
    _MOVE_LEFT = 3
    _MOVE_RIGHT = 4

    # Interaction index constants
    _INTERACT_NONE = 0
    _INTERACT_FIRELINE = 1

    def __init__(self, env, config: dict) -> None:
        """Initialise the heuristic agent.

        Args:
            env: A gymnasium-compatible FireEnv instance.
            config: Configuration dict (unused by heuristic but stored for
                API compatibility).
        """
        super().__init__(env, config)
        # Number of movements (including "none")
        self._num_movements = 5

    def get_action(
        self,
        obs: np.ndarray,
        fire_map: Optional[np.ndarray] = None,
    ) -> int:
        """Return a heuristic action based on the current fire map.

        If ``fire_map`` is not provided, the fire_map channel is extracted
        from the first channel of ``obs`` (channel 0 by convention in FireEnv).
        Note that the obs is normalised, so burning cells will have value
        ``1 / sim_agent_id`` rather than ``1``; the heuristic simply finds
        the maximum value in the channel to locate burning cells when a raw
        fire_map is not available.

        Args:
            obs: Current observation array (H, W, C) float32.
            fire_map: Optional raw (un-normalised) 2D fire map array. When
                supplied the heuristic uses exact BurnStatus values.

        Returns:
            Integer action index in [0, action_space.n).
        """
        # --- Determine agent position ---
        # Try to get position from environment attribute if available.
        try:
            agent_row, agent_col = self.env.agent_pos[0], self.env.agent_pos[1]
        except AttributeError:
            agent_row, agent_col = 0, 0

        # --- Locate burning cells ---
        if fire_map is not None:
            # Raw fire_map: burning cells have value == 1
            burning_positions = np.argwhere(fire_map == 1)
        else:
            # Normalised obs channel 0: find cells with highest values
            # (burning value will be > 0 and typically the highest non-agent value)
            fire_channel = obs[..., 0]
            # Use a threshold of 0.1 to find active burning cells
            burning_positions = np.argwhere(fire_channel > 0.1)

        if len(burning_positions) == 0:
            # No fire detected: stay and do nothing
            return self._MOVE_NONE

        # --- Find nearest burning cell (Manhattan distance) ---
        distances = np.abs(burning_positions[:, 0] - agent_row) + \
                    np.abs(burning_positions[:, 1] - agent_col)
        nearest_idx = int(np.argmin(distances))
        target_row, target_col = burning_positions[nearest_idx]

        # --- Choose movement toward target ---
        dr = target_row - agent_row
        dc = target_col - agent_col

        if abs(dr) >= abs(dc):
            movement_idx = self._MOVE_DOWN if dr > 0 else self._MOVE_UP
        else:
            movement_idx = self._MOVE_RIGHT if dc > 0 else self._MOVE_LEFT

        # --- Always interact with fireline (index 1) ---
        interaction_idx = self._INTERACT_FIRELINE

        # --- Encode action ---
        action = interaction_idx * self._num_movements + movement_idx
        return action

    def train(
        self,
        total_timesteps: int,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        """Heuristic agents do not learn; raises NotImplementedError.

        Raises:
            NotImplementedError: Always. Use a DQNAgent or PPOAgent for learning.
        """
        raise NotImplementedError(
            "HeuristicAgent does not support training. "
            "Use DQNAgent or PPOAgent for learning-based agents."
        )

    def save(self, path: str) -> None:
        """No-op: heuristic agents have no learnable weights to save.

        Args:
            path: Ignored.
        """
        pass

    def load(self, path: str) -> None:
        """No-op: heuristic agents have no weights to load.

        Args:
            path: Ignored.
        """
        pass
