"""Reward functions for the firefighting RL environment.

Two reward systems:

  A. Data-driven env (DataDrivenFireEnv):

     Per AGENT STEP (individual, computed inside the env):
       1. Interception placement — reward depends on t_ahead = fire_arrival[r,c]
          minus fire_timestep when mitigation is placed on an UNBURNED cell.
          GATED: bonus only awarded when agent is within _PLACEMENT_FIRE_RADIUS
          cells of a BURNING cell or fire_ahead_1 cell (prevents station camping).
            t_ahead == 2 → +400  (optimal: 6 h ahead, time to build a full line)
            t_ahead == 1 → +200  (reactive: 3 h ahead, barely in time)
            t_ahead in {3,4} → +150  (proactive positioning)
            near burning/ahead_1-2 → +50  (fallback defensive placement)
            not near fire → −20  (wasted: placing mitigation far from front)
       2. Individual contact bonus — +300/+150/+75 per fire-adjacent edge for
          fireline/scratchline/wetline, routed ONLY to the placing agent.
          Held in pending_agent_rewards; consumed on that agent's next turn.
       3. Individual blocked-cells reward — +600 per cell with fire_arrival ==
          fire_timestep that still has mitigation at the timestep boundary,
          routed to the placing agent.
       4. Step penalty (3-tier, per agent): 0 (≤5 cells to fire), −3 (≤20), −8 (>20).
          Strong gradient driving all agents toward the fire front.
       5. Per-station growth penalty — −150 per newly burned cell that falls in
          the agent's own Voronoi sector (closest to their home station).
          Fire burning outside your sector does NOT penalise you — only fire
          you were responsible for blocking counts against you.

     Sector approach reward REMOVED: Voronoi sectors were pulling northern/SE
     agents to home regions far from the real fire front.  The 3-tier step
     penalty provides the necessary directional gradient without misdirection.

     No population penalty (removed).
     No shared round reward — all signals are individually attributed.

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

# 3-tier timestep penalty — drives agents toward fire.
#   ≤ 5 cells from nearest BURNING cell  →  0.0   (in the fight, no cost)
#   ≤ 20 cells                           → -3.0   (en route)
#   > 20 cells                           → -8.0   (too far away)
# Combined with approach reward (+5/cell toward intercept target):
#   Moving toward fire from far: -8 + 5 = -3 net
#   Moving away from fire:       -8 + 0 = -8 net
#   Clear 5-unit gradient toward fire.
_STEP_PENALTY_NEAR: float = 0.0
_STEP_PENALTY_MID: float = -20.0
_STEP_PENALTY_FAR: float = -50.0
_NEAR_THRESHOLD: int = 5
_MID_THRESHOLD: int = 20


def fighting_fire_step_penalty(agent_positions: list, fire_map: np.ndarray) -> float:
    """3-tier timestep penalty based on distance to nearest burning cell.

    Tiers (Manhattan distance to nearest BURNING cell):
      ≤  5 cells →  0.0  (in direct contact — no penalty)
      ≤ 20 cells → -3.0  (actively travelling toward fire)
      > 20 cells → -8.0  (too far from the front)

    Args:
        agent_positions: List containing one [row, col] for the active agent.
        fire_map:        2-D numpy array with BurnStatus values.

    Returns:
        Scalar penalty.  Returns 0.0 if no BURNING cells exist yet.
    """
    burning_coords = np.argwhere(fire_map == _BURNING)
    if len(burning_coords) == 0:
        return _STEP_PENALTY_NEAR

    pos = agent_positions[0] if isinstance(agent_positions[0], (list, tuple, np.ndarray)) else agent_positions
    r, c = int(pos[0]), int(pos[1])

    min_dist = int(np.min(
        np.abs(burning_coords[:, 0] - r) + np.abs(burning_coords[:, 1] - c)
    ))

    if min_dist <= _NEAR_THRESHOLD:
        return _STEP_PENALTY_NEAR
    elif min_dist <= _MID_THRESHOLD:
        return _STEP_PENALTY_MID
    else:
        return _STEP_PENALTY_FAR


def fighting_fire_round_reward(
    fire_map: np.ndarray,
    population_grid: np.ndarray,
    prev_burned_mask: np.ndarray,
) -> float:
    """DEPRECATED — returns 0.0.

    Growth and population penalties have been removed:
      • Population penalty: removed entirely.
      • Growth penalty: moved to per-station individual signals computed
        inside DataDrivenFireEnv.step() and routed via pending_agent_rewards.

    This function is retained only for API compatibility with any callers
    that have not yet been updated.  It will always return 0.0.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Legacy data_driven_reward — kept for reference, no longer used in training
# ---------------------------------------------------------------------------

def data_driven_reward(
    fire_map: np.ndarray,
    population_grid: np.ndarray,
    prev_burned_mask: np.ndarray,
    total_cells: int,
) -> float:
    """Original reward (area proportion + 1M×population). No longer used."""
    burned_mask = (fire_map == _BURNED) | (fire_map == _BURNING)
    proportion_burned = float(np.sum(burned_mask)) / total_cells
    newly_burned = burned_mask & ~prev_burned_mask
    pop_penalty = -1_000_000.0 * float(np.sum(population_grid[newly_burned]))
    return -proportion_burned + pop_penalty


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
