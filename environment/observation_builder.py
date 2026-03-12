"""Observation builder for the firefighting RL environment.

Collects attribute arrays from the simulation interface, optionally merges
in caller-supplied overrides (e.g. a fire map with the agent position already
stamped on it), normalises selected channels, and stacks everything into a
single (H, W, C) float32 array ready for the neural network.
"""
from typing import Dict, List, Optional

import numpy as np


class ObservationBuilder:
    """Assembles normalised multi-channel observations from simulation data.

    Attributes:
        sim_interface: The FireSimInterface instance to pull attribute data from.
        attributes: Ordered list of attribute names defining the channel order.
        normalized_attributes: Subset of attributes to scale to [0, 1].
        min_maxes: Dict mapping attribute name -> {"min": float, "max": float}.
    """

    def __init__(
        self,
        sim_interface,
        attributes: List[str],
        normalized_attributes: List[str],
        min_maxes: Dict[str, Dict[str, float]],
    ) -> None:
        """Initialise the ObservationBuilder.

        Args:
            sim_interface: An instance of FireSimInterface (or compatible API).
            attributes: Ordered list of channel names to include in the
                observation. The channel order in the output array follows this
                list exactly.
            normalized_attributes: Names of channels that should be scaled to
                [0, 1]. Must be a subset of ``attributes``.
            min_maxes: Per-attribute min/max bounds used for normalisation,
                e.g. {"elevation": {"min": 100.0, "max": 300.0}}.
        """
        self.sim_interface = sim_interface
        self.attributes = attributes
        self.normalized_attributes = normalized_attributes
        self.min_maxes = min_maxes

    def build(self, additional_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Build the (H, W, C) observation array.

        Args:
            additional_data: Optional dict mapping attribute names to 2D numpy
                arrays that override the values returned by the simulation for
                those channels. Typical usage: pass a fire map that already has
                the agent's position stamped on it.

        Returns:
            Float32 array of shape (H, W, len(attributes)) with values in
            [0, 1] for normalised channels and raw simulation values for
            non-normalised channels.
        """
        # Fetch live attribute data from the simulation.
        attr_data = dict(self.sim_interface.get_attribute_data())

        # Override with caller-supplied values (e.g. agent-stamped fire map).
        if additional_data:
            attr_data.update(additional_data)

        # Normalise selected channels in-place.
        for attr in self.normalized_attributes:
            if attr in attr_data and attr in self.min_maxes:
                bounds = self.min_maxes[attr]
                attr_data[attr] = self.normalize(
                    attr_data[attr].astype(np.float32),
                    bounds["min"],
                    bounds["max"],
                )

        # Stack channels in the declared order into a (H, W, C) array.
        channels = []
        for attr in self.attributes:
            channel = np.asarray(attr_data[attr], dtype=np.float32)
            channels.append(channel)

        obs = np.stack(channels, axis=-1).astype(np.float32)
        return obs

    @staticmethod
    def normalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Scale data to [0, 1] using known min/max bounds.

        Args:
            data: Raw numpy array to normalise.
            min_val: Minimum expected value (maps to 0).
            max_val: Maximum expected value (maps to 1).

        Returns:
            Float32 array clipped to [0, 1].
        """
        denom = max_val - min_val
        if denom == 0:
            return np.zeros_like(data, dtype=np.float32)
        normalised = (data - min_val) / denom
        return np.clip(normalised, 0.0, 1.0).astype(np.float32)
