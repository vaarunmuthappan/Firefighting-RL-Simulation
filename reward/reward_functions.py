"""Modular reward functions for the firefighting RL environment.

All functions operate at the module level (not as class methods) so that
individual terms can be combined freely or swapped out in experiments.

Typical usage::

    r = combined_reward(
        fire_map=fire_map,
        screen_size=64,
        is_active=sim_active,
        agent_pos=agent_pos,
        use_proximity=True,
        use_coverage=True,
    )
"""
from typing import List

import numpy as np


def burning_area_penalty(fire_map: np.ndarray, screen_size: int) -> float:
    """Penalise the fraction of the map that is currently burning.

    Args:
        fire_map: 2D numpy array with BurnStatus values. Burning cells have
            value ``1``.
        screen_size: Side length of the square map used to compute total area.

    Returns:
        Reward in range [-10, 0]: ``-(burning_cells / total_cells) * 10``.
    """
    burning = int(np.count_nonzero(fire_map == 1))
    total = screen_size ** 2
    return -(burning / total) * 10.0


def extinguished_bonus(is_active: bool, bonus: float = 10.0) -> float:
    """Return a one-time bonus when the fire is fully extinguished.

    Args:
        is_active: ``True`` if the fire is still spreading, ``False`` if it
            has been fully extinguished.
        bonus: Reward magnitude (default 10.0).

    Returns:
        ``bonus`` if ``not is_active``, else ``0.0``.
    """
    return bonus if not is_active else 0.0


def proximity_penalty(
    agent_pos: List[int],
    fire_map: np.ndarray,
    penalty: float = 2.0,
) -> float:
    """Penalise the agent for standing adjacent to a burning cell.

    Uses 8-connectivity (all 8 neighbours of the agent's position).

    Args:
        agent_pos: [row, col] position of the agent.
        fire_map: 2D numpy array with BurnStatus values.
        penalty: Penalty magnitude (default 2.0, returned as negative).

    Returns:
        ``-penalty`` if any 8-connected neighbour is burning, else ``0.0``.
    """
    row, col = agent_pos[0], agent_pos[1]
    rows, cols = fire_map.shape

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if fire_map[nr][nc] == 1:
                    return -penalty

    return 0.0


def fireline_coverage_bonus(
    fire_map: np.ndarray,
    bonus_per_cell: float = 0.1,
) -> float:
    """Small positive reward for the total number of fireline cells placed.

    Args:
        fire_map: 2D numpy array with BurnStatus values. Fireline cells have
            value ``3``.
        bonus_per_cell: Reward per fireline cell (default 0.1).

    Returns:
        ``fireline_cells * bonus_per_cell``.
    """
    fireline_cells = int(np.count_nonzero(fire_map == 3))
    return fireline_cells * bonus_per_cell


def combined_reward(
    fire_map: np.ndarray,
    screen_size: int,
    is_active: bool,
    agent_pos: List[int],
    use_proximity: bool = False,
    use_coverage: bool = False,
) -> float:
    """Combine reward terms into a single scalar.

    This is the main reward function to call from the environment. The
    burning-area penalty and extinguishment bonus are always included.
    Proximity penalty and fireline coverage bonus are optional.

    Args:
        fire_map: 2D numpy array with BurnStatus values.
        screen_size: Side length of the square map.
        is_active: Whether the fire is still spreading.
        agent_pos: [row, col] agent position (used for proximity penalty).
        use_proximity: Include proximity penalty if ``True``.
        use_coverage: Include fireline coverage bonus if ``True``.

    Returns:
        Combined scalar reward.
    """
    reward = burning_area_penalty(fire_map, screen_size)
    reward += extinguished_bonus(is_active)

    if use_proximity:
        reward += proximity_penalty(agent_pos, fire_map)

    if use_coverage:
        reward += fireline_coverage_bonus(fire_map)

    return reward
