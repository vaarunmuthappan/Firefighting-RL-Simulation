"""Concrete firefighting RL environment wrapping SimFire via FireSimInterface.

This is the main environment class used during training. It implements the
gymnasium Env interface, discrete action decoding, mitigation placement, and
the step-reward calculation described in the project specification.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

from environment.base_env import BaseFireEnv
from environment.observation_builder import ObservationBuilder
from fire_sim.sim_interface import FireSimInterface


class FireEnv(BaseFireEnv):
    """Gymnasium environment for single-agent firefighting using SimFire.

    Action Space
    ------------
    Discrete(len(movements) * len(interactions))

    Decoded as::

        movement    = action % len(self.movements)
        interaction = action // len(self.movements)

    where both ``movements`` and ``interactions`` have "none" prepended at
    index 0.

    Observation Space
    -----------------
    Box(0.0, 1.0, shape=(screen_size, screen_size, num_attrs), dtype=float32)

    Channels are ordered according to the ``attributes`` list in the config.
    The fire_map channel has the agent's position stamped with ``sim_agent_id``.

    Reward
    ------
    Per simulation step: ``-(burning_cells / total_cells) * 10`` (range [-10, 0]).
    On fire extinguished: ``+10`` bonus.
    On non-simulation steps: ``0``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        """Initialise the FireEnv from a merged config dictionary.

        Args:
            config: Flat dict containing keys from env_config.yaml and
                agent_config.yaml merged together. Required keys:
                - screen_size (int)
                - attributes (list[str])
                - normalized_attributes (list[str])
                - movements (list[str])
                - interactions (list[str])
                - agent_speed (int)
                - initial_pos (list[int, int])
                - randomize_pos (bool)
                - max_episode_steps (int)
                - wind_speed, wind_direction, moisture, fire_position_type,
                  max_fire_duration (passed through to sim)
        """
        super().__init__()

        # --- read config values ---
        self.screen_size: int = config.get("screen_size", 64)
        self.attributes: List[str] = list(config.get("attributes", [
            "fire_map", "elevation", "w_0", "sigma", "delta", "M_x"
        ]))
        # Start from config-provided normalised attributes and always include
        # fire_map so that its values (0 .. sim_agent_id) are scaled to [0, 1].
        _norm_attrs: List[str] = list(config.get("normalized_attributes", [
            "elevation", "w_0", "sigma", "delta", "M_x"
        ]))
        if "fire_map" not in _norm_attrs:
            _norm_attrs = ["fire_map"] + _norm_attrs
        self.normalized_attributes: List[str] = _norm_attrs
        self.agent_speed: int = config.get("agent_speed", 9)
        self.initial_pos: List[int] = list(config.get("initial_pos", [15, 15]))
        self.randomize_pos: bool = config.get("randomize_pos", True)
        self.max_episode_steps: int = config.get("max_episode_steps", 2000)

        # movements/interactions — "none" will be prepended below
        movements: List[str] = list(config.get("movements", ["up", "down", "left", "right"]))
        interactions: List[str] = list(config.get("interactions", ["fireline"]))

        # Prepend "none" option (index 0 means do-nothing)
        self.movements: List[str] = ["none"] + movements
        self.interactions: List[str] = ["none"] + interactions

        # sim_agent_id: one above the highest BurnStatus value the sim can return
        # Base values: 0=UNBURNED, 1=BURNING, 2=BURNED, 3=FIRELINE, …
        # then one entry per non-"none" interaction, then +1 for the agent
        self.sim_agent_id: int = 3 + len(self.interactions)  # = 3 + len(interactions_with_none)

        # --- build SimFire ---
        sim_config = FireSimInterface.build_config_dict(
            screen_size=self.screen_size,
            wind_speed=config.get("wind_speed", 2),
            wind_direction=config.get("wind_direction", 135.0),
            moisture=config.get("moisture", 0.03),
            fire_position_type=config.get("fire_position_type", "random"),
            terrain_type=config.get("terrain_type", "functional"),
            max_fire_duration=config.get("max_fire_duration", 4),
            pixel_scale=config.get("pixel_scale", 12),
        )
        self.sim = FireSimInterface(sim_config)

        # Collect bounds for normalisation (fire_map bounds set manually)
        raw_bounds = dict(self.sim.get_attribute_bounds())
        self.min_maxes: Dict[str, Dict[str, float]] = {}
        for attr in self.attributes:
            if attr == "fire_map":
                self.min_maxes[attr] = {"min": 0, "max": float(self.sim_agent_id)}
            elif attr in raw_bounds:
                self.min_maxes[attr] = {
                    "min": float(raw_bounds[attr]["min"]),
                    "max": float(raw_bounds[attr]["max"]),
                }
            else:
                self.min_maxes[attr] = {"min": 0.0, "max": 1.0}

        # --- observation / action spaces ---
        num_attrs = len(self.attributes)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.screen_size, self.screen_size, num_attrs),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(
            len(self.movements) * len(self.interactions)
        )

        # --- observation builder ---
        self.obs_builder = ObservationBuilder(
            sim_interface=self.sim,
            attributes=self.attributes,
            normalized_attributes=self.normalized_attributes,
            min_maxes=self.min_maxes,
        )

        # --- gymnasium spec ---
        self.spec = EnvSpec(
            id="FireEnv-v0",
            entry_point="environment.fire_env:FireEnv",
            max_episode_steps=self.max_episode_steps,
        )

        # internal state (initialised properly in reset)
        self.agent_pos: List[int] = copy.copy(self.initial_pos)
        self.num_steps: int = 0
        self._is_active: bool = True

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to a fresh episode.

        Args:
            seed: Optional RNG seed.
            options: Unused; present for API compatibility.

        Returns:
            obs: Initial observation (H, W, C) float32.
            info: Empty dict.
        """
        super().reset(seed=seed)

        # Reset the simulation
        self.sim.reset()

        # Set agent starting position
        if self.randomize_pos:
            rng = np.random.default_rng(seed)
            self.agent_pos = list(
                rng.integers(0, self.screen_size, size=2, dtype=int).tolist()
            )
        else:
            self.agent_pos = copy.copy(self.initial_pos)

        # Place agent in sim (SimFire expects (col, row, agent_id))
        self.sim.update_agent_position(
            self.agent_pos[1], self.agent_pos[0], 0
        )

        self.num_steps = 0
        self._is_active = True

        # Build initial fire map with agent position stamped on it
        initial_fire_map = np.copy(self.sim.fire_map)
        initial_fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id

        obs = self._build_observation(fire_map_override=initial_fire_map)
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one agent step.

        Args:
            action: Integer in [0, action_space.n).

        Returns:
            obs: New observation (H, W, C) float32.
            reward: Scalar reward for this step.
            terminated: True if fire is extinguished.
            truncated: True if max_episode_steps exceeded.
            info: Dict with "num_steps" key.
        """
        # --- decode action ---
        movement_idx = action % len(self.movements)
        interaction_idx = action // len(self.movements)

        movement_str = self.movements[movement_idx]
        interaction_str = self.interactions[interaction_idx]

        # --- move agent (boundary-clamped) ---
        pos = copy.copy(self.agent_pos)
        if movement_str == "up":
            pos[0] = max(0, pos[0] - 1)
        elif movement_str == "down":
            pos[0] = min(self.screen_size - 1, pos[0] + 1)
        elif movement_str == "left":
            pos[1] = max(0, pos[1] - 1)
        elif movement_str == "right":
            pos[1] = min(self.screen_size - 1, pos[1] + 1)
        # "none" → no change

        self.agent_pos = pos

        # --- mitigation: only on unburned cells ---
        current_fire_map = self.sim.fire_map
        cell_value = current_fire_map[self.agent_pos[0]][self.agent_pos[1]]
        is_empty = (cell_value == 0)

        if is_empty and interaction_str != "none":
            # SimFire mitigation update: (col, row, mitigation_type)
            self.sim.apply_mitigation(self.agent_pos[1], self.agent_pos[0], 3)

        # --- update agent position display in sim ---
        self.sim.update_agent_position(self.agent_pos[1], self.agent_pos[0], 0)

        # --- advance fire simulation every agent_speed steps ---
        reward = 0.0
        if self.num_steps % self.agent_speed == 0:
            fire_map, is_active = self.sim.step()
            fire_map = np.copy(fire_map)
            reward = self._calculate_reward(fire_map)
        else:
            is_active = True
            fire_map = np.copy(self.sim.fire_map)

        # --- stamp agent on fire map ---
        fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id

        # --- extinguishment bonus ---
        if not is_active:
            reward += 10.0

        self._is_active = is_active

        # --- build observation (override fire_map with agent-stamped version) ---
        obs = self._build_observation(fire_map_override=fire_map)

        # --- step bookkeeping ---
        self.num_steps += 1
        truncated = self.num_steps >= self.max_episode_steps
        terminated = not is_active

        return obs, reward, terminated, truncated, {"num_steps": self.num_steps}

    def render(self) -> None:
        """Enable pygame rendering for the current episode."""
        self.sim.rendering = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_reward(self, fire_map: np.ndarray) -> float:
        """Compute step reward based on burning area fraction.

        Args:
            fire_map: 2D array with BurnStatus values (agent may be stamped).

        Returns:
            Reward in range [-10, 0].
        """
        burning = np.count_nonzero(fire_map == 1)
        total = self.screen_size ** 2
        return -(burning / total) * 10.0

    def _build_observation(
        self, fire_map_override: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Assemble the normalised (H, W, C) observation.

        Args:
            fire_map_override: If provided, replaces the fire_map channel with
                this array (e.g. to include the agent's stamped position).

        Returns:
            Float32 array of shape (screen_size, screen_size, num_attrs).
        """
        additional: Optional[Dict[str, np.ndarray]] = None
        if fire_map_override is not None:
            additional = {"fire_map": fire_map_override}
        return self.obs_builder.build(additional_data=additional)
