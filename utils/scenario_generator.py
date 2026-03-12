"""Scenario generation utilities for reproducible evaluation.

Generates fixed sets of (fire_start, agent_start) pairs so that evaluation
runs can be compared fairly across different agents and checkpoints.
"""
from typing import List, Tuple

import numpy as np


def generate_eval_scenarios(
    num_scenarios: int,
    screen_size: int,
    seed: int = 42,
) -> List[dict]:
    """Generate a list of deterministic evaluation scenarios.

    Each scenario specifies an initial fire position and an initial agent
    position. The same seed always produces the same list, enabling
    reproducible comparisons between runs.

    Args:
        num_scenarios: Number of scenarios to generate.
        screen_size: Side length of the square map. Positions are drawn
            from ``[0, screen_size)``.
        seed: NumPy random seed (default 42).

    Returns:
        List of dicts, each with keys:
            - ``"fire_pos"``: ``(x, y)`` tuple for the fire ignition point.
            - ``"agent_pos"``: ``[row, col]`` list for the agent start.

    Example::

        scenarios = generate_eval_scenarios(5, 64)
        # [{"fire_pos": (12, 34), "agent_pos": [56, 7]}, ...]
    """
    rng = np.random.default_rng(seed)
    scenarios: List[dict] = []

    for _ in range(num_scenarios):
        fire_col = int(rng.integers(0, screen_size))
        fire_row = int(rng.integers(0, screen_size))

        agent_row = int(rng.integers(0, screen_size))
        agent_col = int(rng.integers(0, screen_size))

        scenarios.append({
            "fire_pos": (fire_col, fire_row),
            "agent_pos": [agent_row, agent_col],
        })

    return scenarios
