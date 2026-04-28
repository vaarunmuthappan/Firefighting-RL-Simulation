"""Reward functions for the firefighting RL environment.

Two reward systems:

  A. Data-driven env (DataDrivenFireEnv) — fighting_fire_reward():
     Applied per ROUND (once all agents complete their turn):
       1. Mitigation-contact reward: +300/+150/+75 per fireline/scratchline/wetline
          cell edge-adjacent to a burning cell  →  dense causal signal for fire-fighting
       2. Fire-front growth penalty: -50 × new cells ignited this round
          →  immediate signal tied to fire spreading, not lagged cumulative area
       3. Population penalty: -100 × population of newly burned cells
          →  scaled down from -1,000,000 so it doesn't drown the dense signals above
     Applied per AGENT STEP:
       4. Conditional timestep penalty:
            -1   if any agent is within PROXIMITY_RADIUS cells of a burning cell
            -5   otherwise
          →  removes the -10M/episode baseline; being near fire is no longer penalised

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

# Radius (cells) within which an agent is considered "near fire"
PROXIMITY_RADIUS: int = 5

# Mitigation-contact bonuses (per edge-adjacent contact with burning cell)
_MIT_CONTACT_BONUS = {
    _FIRELINE:    300.0,
    _SCRATCHLINE: 150.0,
    _WETLINE:      75.0,
}

# Fire-front growth penalty per newly ignited cell
_GROWTH_PENALTY_PER_CELL: float = 50.0

# Population penalty per person in a newly burned cell
_POP_PENALTY_PER_PERSON: float = 100.0

# Timestep penalty: 3-tier based on proximity to active fire front.
# ≤2 cells: minimal penalty  — agent is right at the fire edge, fully engaged
# ≤5 cells: moderate penalty — agent is near the fire
# >5 cells: large penalty    — agent is far from fire, idling / lost in burned area
_CLOSE_RADIUS:       int   = 2
_STEP_PENALTY_CLOSE: float =  0.0   # ≤2 cells: right at fire — no penalty
_STEP_PENALTY_NEAR:  float = -0.5   # ≤5 cells: near fire
_STEP_PENALTY_FAR:   float = -2.0   # >5 cells: far from fire / idling
# Reduced from (-0.5/-2/-5) so the approach reward (+3/cell) always wins:
# moving 1 cell closer from far away = +3 approach - 2 step = +1 net reward.


def fighting_fire_step_penalty(agent_positions: list, fire_map: np.ndarray) -> float:
    """3-tier conditional timestep penalty applied every agent step.

    Finds the distance from the agent to the nearest BURNING cell and returns:
      ≤2 cells: -0.5  (engaged — right at the fire edge)
      ≤5 cells: -2.0  (near fire)
      >5 cells: -5.0  (far from fire, idling)

    Called with a single-agent position list so each agent gets its own
    individual incentive — no free-rider effect.

    Args:
        agent_positions: List of [row, col] (typically 1 element: current agent).
        fire_map: 2D int array of BurnStatus values.

    Returns:
        Scalar step penalty.
    """
    rows, cols = fire_map.shape
    min_dist = float("inf")
    for pos in agent_positions:
        r, c = int(pos[0]), int(pos[1])
        # Search within the far radius to find the closest burning cell
        r0 = max(0, r - PROXIMITY_RADIUS)
        r1 = min(rows, r + PROXIMITY_RADIUS + 1)
        c0 = max(0, c - PROXIMITY_RADIUS)
        c1 = min(cols, c + PROXIMITY_RADIUS + 1)
        sub_fire = fire_map[r0:r1, c0:c1] == _BURNING
        if np.any(sub_fire):
            fire_rs, fire_cs = np.where(sub_fire)
            fire_rs = fire_rs + r0
            fire_cs = fire_cs + c0
            dists = np.sqrt((fire_rs - r) ** 2.0 + (fire_cs - c) ** 2.0)
            min_dist = min(min_dist, float(dists.min()))

    if min_dist <= _CLOSE_RADIUS:
        return _STEP_PENALTY_CLOSE
    elif min_dist <= PROXIMITY_RADIUS:
        return _STEP_PENALTY_NEAR
    else:
        return _STEP_PENALTY_FAR


def fighting_fire_round_reward(
    fire_map: np.ndarray,
    population_grid: np.ndarray,
    prev_burned_mask: np.ndarray,
) -> float:
    """Round reward applied once per full agent round (after fire advances).

    Combines three components:
      1. Mitigation-contact reward  — dense positive signal for fire-fighting
      2. Fire-front growth penalty  — immediate signal for fire spreading
      3. Population penalty         — scaled to -100×people (not -1M×people)

    Args:
        fire_map:          2D int array of BurnStatus values (post-fire-advance).
        population_grid:   2D float array of population per cell.
        prev_burned_mask:  Boolean mask of cells that were burned/burning
                           BEFORE this fire advance.

    Returns:
        Scalar reward (positive for good fire-fighting, negative for spread).
    """
    # 1. Mitigation-contact reward
    burning_mask = fire_map == _BURNING
    fire_adjacent = binary_dilation(burning_mask, structure=_CROSS_KERNEL)
    contact_reward = 0.0
    for mit_status, bonus in _MIT_CONTACT_BONUS.items():
        contacts = (fire_map == mit_status) & fire_adjacent
        contact_reward += float(np.count_nonzero(contacts)) * bonus

    # 2. Fire-front growth penalty
    burned_now = (fire_map == _BURNED) | (fire_map == _BURNING)
    new_cells = float(np.count_nonzero(burned_now & ~prev_burned_mask))
    growth_penalty = -_GROWTH_PENALTY_PER_CELL * new_cells

    # 3. Population penalty (scaled down)
    newly_burned = burned_now & ~prev_burned_mask
    pop_penalty = -_POP_PENALTY_PER_PERSON * float(np.sum(population_grid[newly_burned]))

    return contact_reward + growth_penalty + pop_penalty


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
