"""Reward functions for the firefighting RL environment.

Two reward systems:

  A. Data-driven env (DataDrivenFireEnv):

     Per AGENT STEP (individual, computed inside the env):
       1. Sector approach reward  — +5 per cell closer to the agent's Voronoi
          sector target.  Each agent's sector is the set of dynamic-k predicted
          fire cells whose Manhattan distance is smallest to that agent's home
          fire station.  This is a station-proximity Voronoi partition: if
          station A is NW and station B is SE, agent A targets the NW portion
          of the predicted fire front and agent B the SE portion.
          Dynamic k = max(1, min(4, ceil(dist_to_fire / actions_per_fire_step))
          so far-away agents aim at a future front and intercept rather than chase.
       2. Interception placement — reward depends on t_ahead = fire_arrival[r,c]
          minus fire_timestep when mitigation is placed on an UNBURNED cell:
            t_ahead == 2 → +400  (optimal: 6 h ahead, time to build a full line)
            t_ahead == 1 → +200  (reactive: 3 h ahead, barely in time)
            t_ahead in {3,4} → +150  (proactive positioning)
            t_ahead >  4 →   0  (fire will arrive — no bonus, no penalty)
            near burning/ahead_1-2 → +50  (fallback defensive placement)
            otherwise → −30  (genuinely wasted: fire never reaches that cell)
       3. Individual contact bonus — +300/+150/+75 per fire-adjacent edge for
          fireline/scratchline/wetline, routed ONLY to the placing agent.
          Held in pending_agent_rewards; consumed on that agent's next turn.
       4. Individual blocked-cells reward — +600 per cell with fire_arrival ==
          fire_timestep that still has mitigation at the timestep boundary,
          routed to the placing agent.
       5. Flat step penalty: −1 per step.
       6. Per-station growth penalty — −50 per newly burned cell that falls in
          the agent's own Voronoi sector (closest to their home station).
          Fire burning outside your sector does NOT penalise you — only fire
          you were responsible for blocking counts against you.

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

# Flat timestep penalty — uniform; the sector approach reward (+5/cell) already
# provides all the proximity gradient needed.
_STEP_PENALTY_FLAT: float = -1.0


def fighting_fire_step_penalty(agent_positions: list, fire_map: np.ndarray) -> float:
    """Flat timestep penalty applied every agent step: −1.0.

    Replaces the old 3-tier proximity-based logic.  The sector approach
    reward (+5 per cell closer to the intercept target) already provides a
    strong directional gradient; the step penalty only needs to discourage
    doing nothing.

    Args:
        agent_positions: Unused (kept for API compatibility).
        fire_map:        Unused (kept for API compatibility).

    Returns:
        -1.0 always.
    """
    return _STEP_PENALTY_FLAT


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
