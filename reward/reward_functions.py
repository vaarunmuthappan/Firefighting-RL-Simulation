"""Reward functions for the firefighting RL environment.

Two reward systems:
  A. Data-driven env (DataDrivenFireEnv):
     1. Area penalty: -proportion of total cells that are burned/burning
     2. Population penalty: -1,000,000 × population of newly burned pixels
     3. Timestep penalty: -1000 per step (applied in env.step())

  B. SimFire env (FireEnv) — kept for backwards compatibility:
     1. Burning RoS penalty
     2. Mitigation contact reward
     3. Timestep penalty: -1000 per step
"""
import numpy as np
from scipy.ndimage import binary_dilation

# BurnStatus values
_UNBURNED = 0
_BURNING = 1
_BURNED = 2
_FIRELINE = 3
_SCRATCHLINE = 4
_WETLINE = 5

# SimFire default RoS attenuation values (ft/min reduction)
_ATTENUATION = {
    _FIRELINE: 980.0,
    _SCRATCHLINE: 490.0,
    _WETLINE: 245.0,
}

# 4-connected structuring element (edge-adjacent, no diagonals)
_CROSS_KERNEL = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=bool)


# ---------------------------------------------------------------------------
# Data-driven reward (DataDrivenFireEnv)
# ---------------------------------------------------------------------------

# def data_driven_reward(
#     fire_map: np.ndarray,
#     population_grid: np.ndarray,
#     prev_burned_mask: np.ndarray,
#     total_cells: int,
# ) -> float:
#     """Reward for the data-driven fire environment.

#     Args:
#         fire_map: 2D array with BurnStatus values.
#         population_grid: 2D float array of population per cell.
#         prev_burned_mask: Boolean 2D array — True for cells that were already
#             burned/burning BEFORE this fire advance.
#         total_cells: Total number of cells in the grid (rows × cols).

#     Returns:
#         Scalar reward (area penalty + population penalty).
#         Timestep penalty is applied separately in env.step().
#     """
#     burned_mask = (fire_map == _BURNED) | (fire_map == _BURNING)
#     proportion_burned = float(np.sum(burned_mask)) / total_cells
#     area_penalty = -proportion_burned

#     newly_burned = burned_mask & ~prev_burned_mask
#     pop_penalty = -1_000_000.0 * float(np.sum(population_grid[newly_burned]))
#     pop_penalty = -100.0 * float(np.sum(population_grid[newly_burned]))
#     return area_penalty + pop_penalty

def data_driven_reward(
    fire_map: np.ndarray,
    population_grid: np.ndarray,
    prev_burned_mask: np.ndarray,
    total_cells: int,
) -> float:
    burned_mask = (fire_map == 2) | (fire_map == 1)
    proportion_burned = float(np.sum(burned_mask)) / total_cells
    area_penalty = -10.0 * proportion_burned

    newly_burned = burned_mask & ~prev_burned_mask
    pop_penalty = -1000.0 * float(np.sum(population_grid[newly_burned]))

    return area_penalty + pop_penalty

# ---------------------------------------------------------------------------
# SimFire reward (FireEnv) — backwards compatible
# ---------------------------------------------------------------------------

def burning_ros_penalty(fire_map: np.ndarray, ros_grid: np.ndarray) -> float:
    """Negative sum of rate-of-spread over all burning cells."""
    burning_mask = (fire_map == _BURNING)
    return -float(np.sum(ros_grid[burning_mask]))


def mitigation_contact_reward(fire_map: np.ndarray) -> float:
    """Positive reward for mitigation cells that share an edge with fire."""
    burning_mask = (fire_map == _BURNING)
    fire_adjacent = binary_dilation(burning_mask, structure=_CROSS_KERNEL)

    reward = 0.0
    for mit_status, attenuation in _ATTENUATION.items():
        mit_mask = (fire_map == mit_status)
        contacts = mit_mask & fire_adjacent
        reward += float(np.count_nonzero(contacts)) * attenuation

    return reward


def combined_reward(fire_map: np.ndarray, ros_grid: np.ndarray) -> float:
    """Compute the full reward for a SimFire simulation step."""
    return burning_ros_penalty(fire_map, ros_grid) + mitigation_contact_reward(fire_map)
