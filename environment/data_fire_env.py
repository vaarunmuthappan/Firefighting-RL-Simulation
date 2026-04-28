"""Data-driven firefighting RL environment using real Camp Fire VIIRS data.

Fire spread follows the precomputed fire_arrival grid derived from VIIRS
satellite observations (FEDS dataset). Fire advances proportionally — cells
are distributed evenly across sub-steps within each fire timestep to avoid
visual/temporal jumps.

Multi-agent support: multiple fire engines start from real fire station
locations. Agents take sequential turns; fire advances after all agents
have acted once.

Observation (LOCAL WINDOW — agent always at centre):
  25×25 window centred on the current agent, 7 channels:
    0: fire_map         — normalised burn status / 5.0  →  [0, 1]
                          (0=unburned, 0.2=burning, 0.4=burned,
                           0.6=fireline, 0.8=scratchline, 1.0=wetline)
    1: fire_ahead_1     — binary: cells burning at fire_timestep+1  (3 h)
    2: fire_ahead_2     — binary: cells burning at fire_timestep+2  (6 h)
    3: fire_ahead_3     — binary: cells burning at fire_timestep+3  (9 h)
    4: fire_ahead_4     — binary: cells burning at fire_timestep+4  (12 h)
    5: other_agents     — binary: 1.0 at every other agent's position
    6: mitigation       — binary: 1.0 where any mitigation type exists

  Using a LOCAL window (vs global downsampled view) is the core fix for
  agent clustering: each agent always appears at window centre, so two
  agents at different positions see DIFFERENT local contexts even when
  the global fire state is identical.  Different inputs → different
  policy outputs → agents naturally spread across fire fronts.

Reward design (all signals individually attributed — no shared/pooled rewards):
  1. Sector approach (+5/cell): each agent targets the centroid of predicted
     fire cells in their home station's Voronoi sector (cells closest to that
     station by Manhattan distance).  Dynamic k = max(1, min(4,
     ceil(dist_to_fire / actions_per_fire_step))) so far agents aim at a
     future front rather than chasing the current one.
  2. Interception placement: reward tier = t_ahead = fire_arrival[r,c] −
     fire_timestep when mitigation is placed.  t+2→+400, t+1→+200,
     t+3/4→+150, t>4→0, near-fire fallback→+50, genuinely wasted→−30.
  3. Individual contact bonus (+300/+150/+75): routed to the agent who placed
     the mitigation cell (mitigation_owner dict), paid next turn.
  4. Individual blocked-cells reward (+600): for cells that fire_arrival maps
     to this timestep but still carry mitigation, paid to the placing agent.
  5. Per-station growth penalty (−50/cell): newly burned cells are assigned
     to the nearest fire station by Voronoi; agents from THAT station receive
     the penalty.  Agents from other stations are not penalised.
  6. Flat step penalty: −1 per step.
"""
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from scipy.ndimage import binary_dilation, distance_transform_edt

from environment.base_env import BaseFireEnv
from fire_sim.sim_interface import FireSimInterface
from reward.reward_functions import (
    fighting_fire_round_reward,
    fighting_fire_step_penalty,
)

# BurnStatus values
UNBURNED = 0
BURNING = 1
BURNED = 2
FIRELINE = 3
SCRATCHLINE = 4
WETLINE = 5

MITIGATION_MAP: Dict[str, int] = {
    "fireline": FIRELINE,
    "scratchline": SCRATCHLINE,
    "wetline": WETLINE,
}

# Observation: full-grid centred window + agent-ID channel.
#
# Two ingredients that together solve the clustering problem:
#
# 1. FULL-GRID CENTRED WINDOW (220×220 raw → 55×55 after 4× downsample)
#    The window is larger than the grid diagonal (~147 cells) so the ENTIRE
#    Camp Fire extent is always visible, regardless of where the agent sits.
#    The agent is always at the centre pixel of the downsampled obs, so agents
#    at different positions see different views: a top-right agent sees the
#    grid in the bottom-left of its window; a bottom-left agent sees it in
#    the top-right.  Different inputs → different outputs → spreading.
#    220 = 2 × 107 + 6 (divisible by 4).  220 ÷ 4 = 55.
#
# 2. AGENT-ID CHANNEL (channel 7, constant per agent)
#    Even when two agents ARE at the same position they would otherwise see
#    identical 7-channel windows and the shared policy would output the same
#    action for both — the clustering deadlock.  Channel 7 is filled with a
#    constant = agent_idx / (num_agents − 1), so:
#       agent 0  → channel-7 = 0.000
#       agent 5  → channel-7 = 0.556
#       agent 9  → channel-7 = 1.000
#    The policy sees a unique "who am I" signal and can learn role-specific
#    behaviour: "ID≈0: cover the left flank; ID≈1: cover the right flank."
#    This breaks the symmetry at the source — no crowding penalty needed.
#
# Final obs shape: (55, 55, 8)  — small enough for fast training, big enough
# to see the whole fire and distinguish every agent.
OBS_WINDOW_SIZE: int = 220   # raw window before downsampling (> grid diagonal)
OBS_DOWNSAMPLE: int = 4      # spatial stride: 220 → 55 pixels
OBS_N_CHANNELS: int = 8      # 7 spatial + 1 agent-ID channel

# ---------------------------------------------------------------------------
# Reward constants — sector approach + interception placement + individual
#                    contact/blocked signals (see reward_functions.py docstring
#                    for full rationale).
# ---------------------------------------------------------------------------

# Sector approach reward: raw per cell moved closer to agent's sector target.
# Net incentive vs flat step penalty (-1): +5 − 1 = +4 per approach step.
_SECTOR_APPROACH_REWARD: float = 5.0

# Interception placement bonuses (raw, pre-reward-scale) by t_ahead buckets.
# t_ahead = fire_arrival[r,c] − fire_timestep
_INTERCEPTION_BONUS: Dict[int, float] = {
    1: 200.0,   # reactive — fire arrives in 1 fire-timestep (3 h)
    2: 400.0,   # optimal  — fire arrives in 2 fire-timesteps (6 h)
    3: 150.0,   # proactive — fire arrives in 3 fire-timesteps (9 h)
    4: 150.0,   # proactive — fire arrives in 4 fire-timesteps (12 h)
}
# Fallback: placement is not on a predicted path but IS near active/predicted fire
_INTERCEPTION_NEAR_BONUS: float = 50.0
# Radius for "near fire" fallback check (cells)
_INTERCEPTION_NEAR_RADIUS: int = 5
# Wasted penalty: fire NEVER reaches this cell AND not near any fire/prediction.
# t_ahead > 4 gets 0 (fire will reach eventually, just beyond the 12h window).
_WASTED_PLACEMENT_PENALTY: float = 30.0

# Individual mitigation contact bonus — routed to the agent who placed the cell.
# Stored in pending_agent_rewards; consumed on the placing agent's next turn.
# Bonuses match the original round-reward values so the agent receives the same
# signal, just attributed to the correct agent.
_MIT_CONTACT_BONUS_INDIV: Dict[int, float] = {
    FIRELINE:    300.0,
    SCRATCHLINE: 150.0,
    WETLINE:      75.0,
}

# Individual blocked-cells reward — higher than old shared value (+300) because
# it is now properly attributed: only the placing agent gets the credit.
_BLOCKED_CELLS_REWARD: float = 600.0

# Per-station growth penalty (raw, pre-reward-scale) per newly burned cell
# that falls in the agent's home station's Voronoi sector.  Fire burning in
# someone else's territory does NOT penalise this agent.
_GROWTH_PENALTY_PER_CELL: float = 50.0

# 4-connected structuring element for fire-adjacency detection
_CROSS_KERNEL = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=bool)

# Mitigation resistance: fire-adjacent sub-steps before degradation.
# FIRELINE is permanent (not listed = no degradation).
_MITIGATION_RESISTANCE: Dict[int, int] = {
    SCRATCHLINE: 108,   # ~9 h (3 fire timesteps)
    WETLINE: 36,        # ~3 h (1 fire timestep)
}

# Spiral offsets for spreading trucks from the same station to unique cells.
_SPREAD_OFFSETS: List[Tuple[int, int]] = [
    (0, 0), (0, 1), (1, 0), (0, -1), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1),
    (0, 2), (2, 0), (0, -2), (-2, 0),
    (1, 2), (2, 1), (-1, 2), (2, -1),
    (-2, 1), (-1, -2), (2, 2), (-2, 2), (-2, -2), (2, -2),
]

# ---------------------------------------------------------------------------
# Time budget: minutes consumed per action within one 3-hour fire timestep.
#
# Each agent carries a time budget of `hours_per_timestep × 60` minutes
# (default 180 min).  Every action deducts from that budget; when the
# budget is exhausted the action is rejected (agent stays put, no placement).
# Budget resets to full at each fire-timestep boundary.
#
# Derivation (NWCG fireline production rates):
#   Road travel  : 375 m ÷ 45 km/h  ≈  5 min / cell
#   Off-road     : 375 m ÷ 25 km/h  ≈ 15 min / cell (navigation overhead)
#   Fireline     : 375 m ÷ 500 m/h  ≈ 45 min / cell (bulldozer rate)
#   Scratchline  : lighter cut       ≈ 30 min / cell
#   Wetline      : spray-based       ≈ 20 min / cell
#
# Verification (180 min budget):
#   Max road moves      : 180 ÷  5 = 36 cells  ✓ (== actions_per_fire_step)
#   Max off-road moves  : 180 ÷ 15 = 12 cells
#   Max fireline cells  : 180 ÷ 45 =  4 cells
#   Mix (drive 10 + 2 fireline) : 10×5 + 2×45 = 140 min  ✓ fits in budget
# ---------------------------------------------------------------------------
_MOVE_TIME_ROAD: float    =  5.0   # minutes per cell on a road cell
_MOVE_TIME_OFFROAD: float = 15.0   # minutes per cell off-road

# Build time by mitigation BurnStatus value (minutes per cell)
_BUILD_TIME: Dict[int, float] = {
    FIRELINE:    45.0,
    SCRATCHLINE: 30.0,
    WETLINE:     20.0,
}


class DataDrivenFireEnv(BaseFireEnv):
    """Gymnasium environment using real fire spread data from VIIRS observations.

    Fire progresses deterministically based on the precomputed fire_arrival
    grid.  Each fire timestep (3 hours), newly burning cells are distributed
    proportionally across agent sub-steps so fire expands gradually.

    Multiple agents (fire engines) start at real fire station locations.
    Each agent takes a sequential turn; fire advances after all agents
    have acted once (one complete round).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.screen_size: int = config.get("screen_size", 250)

        # ---- Load fire arrival data ----
        data_dir = Path(config.get("data_dir", "data"))
        arrival_path = data_dir / config.get(
            "fire_arrival_file", "camp_fire_full_arrival_375m.npy"
        )
        meta_path = data_dir / config.get(
            "fire_meta_file", "camp_fire_full_meta.json"
        )

        self.fire_arrival = np.load(arrival_path)
        with open(meta_path) as f:
            self.fire_meta = json.load(f)

        grid_rows = self.fire_meta.get("grid_rows", self.fire_arrival.shape[0])
        grid_cols = self.fire_meta.get("grid_cols", self.fire_arrival.shape[1])
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        assert self.fire_arrival.shape == (grid_rows, grid_cols), (
            f"fire_arrival shape {self.fire_arrival.shape} != ({grid_rows}, {grid_cols})"
        )

        self.max_fire_timestep = int(self.fire_arrival.max())
        self.hours_per_timestep = self.fire_meta.get("hours_per_timestep", 3.0)
        self.time_budget = self.hours_per_timestep * 60.0  # minutes per fire timestep

        # ---- Load roads grid ----
        roads_path = data_dir / config.get("roads_file", "camp_fire_roads_375m.npy")
        if roads_path.exists():
            self.roads_grid = np.load(roads_path)
        else:
            self.roads_grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)

        # ---- Load population grid ----
        pop_path = data_dir / config.get(
            "population_file", "camp_fire_population_375m.npy"
        )
        if pop_path.exists():
            self.population_grid = np.load(pop_path).astype(np.float32)
        else:
            self.population_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        # ---- Load fire stations ----
        stations_path = data_dir / config.get(
            "stations_file", "camp_fire_stations.json"
        )
        if stations_path.exists():
            with open(stations_path) as f:
                all_stations = json.load(f)
            self.stations = [s for s in all_stations if s.get("in_grid", False)]
        else:
            self.stations = []

        # If no stations in grid, create entry points from nearest out-of-grid stations
        if not self.stations and stations_path.exists():
            with open(stations_path) as f:
                all_stations = json.load(f)
            self.stations = self._create_entry_stations(all_stations)

        # ---- Agent / action config ----
        self.initial_pos: List[int] = list(config.get("initial_pos", [15, 15]))
        self.max_episode_steps: int = config.get("max_episode_steps", 5000)
        self.reward_scale: float = float(config.get("reward_scale", 1.0))

        movements: List[str] = list(
            config.get("movements", ["up", "down", "left", "right"])
        )
        interactions: List[str] = list(config.get("interactions", ["fireline"]))
        self.movements: List[str] = ["none"] + movements
        self.interactions: List[str] = ["none"] + interactions

        max_mitigation = max(
            (MITIGATION_MAP.get(name, FIRELINE) for name in interactions),
            default=FIRELINE,
        )
        self.sim_agent_id: int = max_mitigation + 2

        self.num_agents = sum(s.get("trucks", 1) for s in self.stations)
        if self.num_agents == 0:
            self.num_agents = 1  # fallback

        # Precompute station grid positions as a (num_stations, 2) int array.
        # Used by _get_sector_target for Voronoi partitioning and by the
        # per-station growth penalty to attribute fire spread to a responsible agent.
        self.station_positions: np.ndarray = np.array(
            [[s["grid_row"], s["grid_col"]] for s in self.stations],
            dtype=np.int32,
        ) if self.stations else np.zeros((1, 2), dtype=np.int32)

        self.actions_per_agent = len(self.movements) * len(self.interactions)

        # ---- Pre-compute proportional fire spread schedule ----
        self._precompute_fire_schedule(config)

        # ---- Build SimFire interface ----
        # Retained for terrain data used by build_terrain_base_rgb() and
        # any future observation channels.  Not used in _build_observation().
        sim_config = FireSimInterface.build_config_dict(
            screen_size=max(grid_rows, grid_cols),
            terrain_type=config.get("terrain_type", "operational"),
            latitude=config.get(
                "latitude", self.fire_meta.get("lat_min", 39.58)
            ),
            longitude=config.get(
                "longitude", self.fire_meta.get("lon_min", -121.79)
            ),
            resolution=config.get("resolution", 30),
            landfire_year=config.get("landfire_year", 2020),
            pixel_scale=config.get("pixel_scale", 12),
            ros_attenuation=config.get("ros_attenuation", True),
            wind_speed=config.get("wind_speed", 2),
            wind_direction=config.get("wind_direction", 135.0),
            moisture=config.get("moisture", 0.03),
            fire_position_type="static",
            max_fire_duration=100,
        )
        self.sim = FireSimInterface(sim_config)

        # ---- Observation space ----
        # Full-grid centred window + agent-ID channel (see module constants).
        import math
        self.obs_window_size: int = config.get("obs_window_size", OBS_WINDOW_SIZE)
        self.obs_downsample: int = config.get("obs_downsample", OBS_DOWNSAMPLE)
        # Output size after stride-downsampling: window[::d, ::d] gives ceil(win/d)
        self.obs_out_size: int = math.ceil(self.obs_window_size / self.obs_downsample)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_out_size, self.obs_out_size, OBS_N_CHANNELS),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.actions_per_agent)

        self.spec = EnvSpec(
            id="DataFireEnv-v0",
            entry_point="environment.data_fire_env:DataDrivenFireEnv",
            max_episode_steps=self.max_episode_steps,
        )

        # ---- State (properly initialised in reset()) ----
        self.agents: List[Dict[str, Any]] = []
        self.current_agent_idx: int = 0
        self.num_steps: int = 0
        self.fire_timestep: int = 0
        self.fire_sub_step: int = 0
        self.fire_map: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=int)
        self.prev_burned_mask: np.ndarray = np.zeros(
            (grid_rows, grid_cols), dtype=bool
        )
        self._is_active: bool = True
        self.mitigation_exposure: Dict[Tuple[int, int], int] = {}
        # Ownership: maps each mitigation cell (r,c) → agent_idx who placed it.
        # Used to route contact bonuses and blocked-cells rewards to the correct agent.
        self.mitigation_owner: Dict[Tuple[int, int], int] = {}
        # Per-agent pending rewards (contact + blocked) accumulated during round
        # processing and paid out on each agent's next turn.
        self.pending_agent_rewards: List[float] = [0.0] * max(self.num_agents, 1)

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_fire_schedule(self, config: dict) -> None:
        """Pre-compute which cells ignite at each sub-step for proportional spread.

        For each fire timestep t, cells with arrival==t are sorted by distance
        from the t-1 fire perimeter and distributed evenly across sub-steps.
        """
        self.actions_per_fire_step = config.get("actions_per_fire_step", 36)
        self.fire_schedule: Dict[int, List[List[Tuple[int, int]]]] = {}

        cumulative_fire = np.zeros(
            (self.fire_arrival.shape[0], self.fire_arrival.shape[1]), dtype=bool
        )

        for t in range(0, self.max_fire_timestep + 1):
            cell_coords = np.argwhere(self.fire_arrival == t)
            if len(cell_coords) == 0:
                self.fire_schedule[t] = []
                continue

            if t == 0 or not np.any(cumulative_fire):
                self.fire_schedule[t] = [list(map(tuple, cell_coords))]
            else:
                dist = distance_transform_edt(~cumulative_fire)
                distances = dist[cell_coords[:, 0], cell_coords[:, 1]]
                order = np.argsort(distances)
                cell_coords = cell_coords[order]

                n_sub = self.actions_per_fire_step
                batches: List[List[Tuple[int, int]]] = []
                if len(cell_coords) <= n_sub:
                    for cell in cell_coords:
                        batches.append([(int(cell[0]), int(cell[1]))])
                else:
                    splits = np.array_split(cell_coords, n_sub)
                    for split in splits:
                        batches.append([(int(rr), int(cc)) for rr, cc in split])

                self.fire_schedule[t] = batches

            cumulative_fire[cell_coords[:, 0], cell_coords[:, 1]] = True

    def _create_entry_stations(self, all_stations: list) -> list:
        """Create virtual entry stations from nearest out-of-grid stations."""
        center_lat = (self.fire_meta["lat_min"] + self.fire_meta["lat_max"]) / 2
        center_lon = (self.fire_meta["lon_min"] + self.fire_meta["lon_max"]) / 2

        out_of_grid = [s for s in all_stations if not s.get("in_grid", False)]
        if not out_of_grid:
            return [
                {
                    "name": "default",
                    "grid_row": 0,
                    "grid_col": 0,
                    "trucks": 1,
                    "in_grid": True,
                }
            ]

        for s in out_of_grid:
            s["_dist"] = (
                (s["lat"] - center_lat) ** 2 + (s["lon"] - center_lon) ** 2
            ) ** 0.5

        out_of_grid.sort(key=lambda s: s["_dist"])
        nearest = out_of_grid[:2]

        entry_stations = []
        for s in nearest:
            dlat = s["lat"] - center_lat
            dlon = s["lon"] - center_lon

            if abs(dlat) > abs(dlon):
                entry_row = 0 if dlat > 0 else self.grid_rows - 1
                col_frac = (s["lon"] - self.fire_meta["lon_min"]) / (
                    self.fire_meta["lon_max"] - self.fire_meta["lon_min"]
                )
                entry_col = max(
                    0, min(self.grid_cols - 1, int(col_frac * self.grid_cols))
                )
            else:
                entry_col = 0 if dlon < 0 else self.grid_cols - 1
                row_frac = (self.fire_meta["lat_max"] - s["lat"]) / (
                    self.fire_meta["lat_max"] - self.fire_meta["lat_min"]
                )
                entry_row = max(
                    0, min(self.grid_rows - 1, int(row_frac * self.grid_rows))
                )

            if self.roads_grid is not None:
                entry_row, entry_col = self._snap_to_road(entry_row, entry_col)

            entry_stations.append(
                {
                    "name": f"Entry from {s['name']}",
                    "grid_row": entry_row,
                    "grid_col": entry_col,
                    "trucks": s.get("trucks", 1),
                    "in_grid": True,
                }
            )

        return entry_stations

    def _snap_to_road(
        self, row: int, col: int, search_radius: int = 10
    ) -> Tuple[int, int]:
        """Find the nearest road cell to (row, col)."""
        best_r, best_c = row, col
        best_dist = float("inf")
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    if self.roads_grid[r, c] > 0:
                        d = dr * dr + dc * dc
                        if d < best_dist:
                            best_dist = d
                            best_r, best_c = r, c
        return best_r, best_c

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.sim.reset()
        self.mitigation_exposure = {}
        self.mitigation_owner = {}
        self.pending_agent_rewards = [0.0] * max(self.num_agents, 1)

        self.agents = []
        occupied: set = set()
        for s_idx, station in enumerate(self.stations):
            base_r, base_c = station["grid_row"], station["grid_col"]
            for truck_idx in range(station.get("trucks", 1)):
                pos = None
                for dr, dc in _SPREAD_OFFSETS:
                    rr = max(0, min(self.grid_rows - 1, base_r + dr))
                    cc = max(0, min(self.grid_cols - 1, base_c + dc))
                    if (rr, cc) not in occupied:
                        pos = [rr, cc]
                        occupied.add((rr, cc))
                        break
                if pos is None:
                    pos = [base_r, base_c]
                self.agents.append(
                    {
                        "pos": pos,
                        "station": station.get("name", "unknown"),
                        "station_idx": s_idx,       # index into self.stations / station_positions
                        "truck_id": truck_idx,
                        "time_remaining": self.time_budget,
                    }
                )

        if not self.agents:
            self.agents = [
                {
                    "pos": copy.copy(self.initial_pos),
                    "station": "default",
                    "truck_id": 0,
                    "time_remaining": self.time_budget,
                }
            ]
            self.num_agents = 1

        self.current_agent_idx = 0
        self.num_steps = 0
        self.fire_timestep = 0
        self.fire_sub_step = 0
        self._is_active = True

        self.fire_map = np.full(
            (self.grid_rows, self.grid_cols), UNBURNED, dtype=int
        )
        self.prev_burned_mask = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=bool
        )

        self._ignite_full_timestep(0)

        self.agent_pos = self.agents[0]["pos"]
        obs = self._build_observation()
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        agent = self.agents[self.current_agent_idx]

        # Decode action
        movement_idx = action % len(self.movements)
        interaction_idx = action // len(self.movements)
        movement_str = self.movements[movement_idx]
        interaction_str = self.interactions[interaction_idx]

        # ---- Compute proposed destination ----
        _old_pos = list(agent["pos"])
        new_pos = list(agent["pos"])
        if movement_str == "up":
            new_pos[0] = max(0, new_pos[0] - 1)
        elif movement_str == "down":
            new_pos[0] = min(self.grid_rows - 1, new_pos[0] + 1)
        elif movement_str == "left":
            new_pos[1] = max(0, new_pos[1] - 1)
        elif movement_str == "right":
            new_pos[1] = min(self.grid_cols - 1, new_pos[1] + 1)

        # ---- Time-budget check -----------------------------------------------
        # Calculate the total minutes this action would consume:
        #   movement cost : road cell → _MOVE_TIME_ROAD, off-road → _MOVE_TIME_OFFROAD
        #   build cost    : depends on mitigation type (_BUILD_TIME dict)
        # If the agent has insufficient time remaining the action is REJECTED:
        # the agent stays put, no mitigation is placed, and the step penalty is
        # still applied (being out of time does not make the fire wait).
        _move_cost: float = 0.0
        if movement_str != "none" and new_pos != _old_pos:
            _dest_on_road = (
                self.roads_grid[new_pos[0], new_pos[1]] > 0
                if self.roads_grid is not None
                else False
            )
            _move_cost = _MOVE_TIME_ROAD if _dest_on_road else _MOVE_TIME_OFFROAD

        _build_cost: float = 0.0
        if interaction_str != "none":
            _mit_type = MITIGATION_MAP.get(interaction_str, FIRELINE)
            _build_cost = _BUILD_TIME.get(_mit_type, 45.0)

        _time_cost = _move_cost + _build_cost
        _budget_exceeded = _time_cost > 0 and agent["time_remaining"] < _time_cost

        if _budget_exceeded:
            # Reject: undo proposed movement, skip placement
            new_pos = list(_old_pos)
            interaction_str = "none"
        else:
            agent["time_remaining"] -= _time_cost

        # ---- Commit movement ----
        if new_pos != list(agent["pos"]):
            agent["pos"] = new_pos

        # Drain any pending individual rewards accumulated for this agent
        # during previous round processing (contact bonuses + blocked-cells).
        reward: float = self.pending_agent_rewards[self.current_agent_idx]
        self.pending_agent_rewards[self.current_agent_idx] = 0.0

        r, c = agent["pos"]
        _old_r, _old_c = _old_pos[0], _old_pos[1]

        # Pre-slice fire arrival grid to grid bounds (reused multiple times below)
        arr_full = self.fire_arrival[: self.grid_rows, : self.grid_cols]

        # ---- Sector approach reward ------------------------------------------------
        # Each agent targets a unique ROW-SECTOR of the dynamic-k predicted fire front.
        # Dynamic k: k = max(1, min(4, ceil(dist_to_nearest_burning / actions_per_step)))
        # so distant agents target farther-ahead fronts and intercept rather than chase.
        #
        # Sector assignment: sort all fire_ahead_k cells by row, slice evenly into N
        # segments, agent i → segment i.  Two agents at the same cell get pulled in
        # opposite directions (agent 0 → north slice, agent N-1 → south slice).
        # This geometrically breaks clustering without any explicit crowding penalty.
        if [r, c] != [_old_r, _old_c]:
            _target = self._get_sector_target(self.current_agent_idx, r, c)
            if _target is not None:
                _tr, _tc = _target
                _new_dist = abs(r - _tr) + abs(c - _tc)
                _old_dist = abs(_old_r - _tr) + abs(_old_c - _tc)
                _delta = _old_dist - _new_dist  # positive = moved closer
                if _delta > 0:
                    reward += _delta * _SECTOR_APPROACH_REWARD

        # ---- Interception placement reward ----------------------------------------
        # Applied only when placing mitigation on an UNBURNED cell.
        # Reward tier depends on t_ahead = fire_arrival[r,c] − fire_timestep:
        #   t_ahead == 2 → +400  optimal intercept (ahead of fire, time to build line)
        #   t_ahead == 1 → +200  reactive (barely in time)
        #   t_ahead in {3,4} → +150  proactive (early positioning)
        #   t_ahead >  4 →   0   fire will arrive eventually — no bonus, no penalty
        #   near burning/ahead → +50  fallback: defensive near-fire placement
        #   otherwise → −30  genuinely wasted: fire never reaches AND not near fire
        cell_value = self.fire_map[r, c]
        if cell_value == UNBURNED and interaction_str != "none":
            mitigation_type = MITIGATION_MAP.get(interaction_str, FIRELINE)
            self.fire_map[r, c] = mitigation_type
            self.mitigation_owner[(r, c)] = self.current_agent_idx  # ownership

            t_ahead = int(arr_full[r, c]) - self.fire_timestep

            if t_ahead in _INTERCEPTION_BONUS:
                reward += _INTERCEPTION_BONUS[t_ahead]
            elif t_ahead > 4:
                pass  # fire will reach eventually — no bonus, no penalty
            else:
                # fire_arrival <= fire_timestep (won't burn) or 0 (origin cell).
                # Grant near-fire fallback if within proximity of active/predicted fire.
                _pr0 = max(0, r - _INTERCEPTION_NEAR_RADIUS)
                _pr1 = min(self.grid_rows, r + _INTERCEPTION_NEAR_RADIUS + 1)
                _pc0 = max(0, c - _INTERCEPTION_NEAR_RADIUS)
                _pc1 = min(self.grid_cols, c + _INTERCEPTION_NEAR_RADIUS + 1)
                _loc_fire = self.fire_map[_pr0:_pr1, _pc0:_pc1]
                _loc_arr = arr_full[_pr0:_pr1, _pc0:_pc1]
                _near_fire = (
                    np.any(_loc_fire == BURNING)
                    or np.any(_loc_arr == self.fire_timestep + 1)
                    or np.any(_loc_arr == self.fire_timestep + 2)
                )
                if _near_fire:
                    reward += _INTERCEPTION_NEAR_BONUS
                else:
                    reward -= _WASTED_PLACEMENT_PENALTY

        # ---- Advance to next agent ----
        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents

        # ---- When all agents have acted: advance fire + compute round signals ----
        if self.current_agent_idx == 0:
            # Snapshot burned state BEFORE fire advances (for delta signals)
            self.prev_burned_mask = (self.fire_map == BURNED) | (
                self.fire_map == BURNING
            )

            # Advance fire by one sub-step
            self._advance_fire_substep()

            # ---- Individual contact bonus (per owner) ----------------------------
            # After fire advances, determine which mitigation cells are now adjacent
            # to burning fire and credit the placing agent — not all agents.
            # Stored in pending_agent_rewards; paid on that agent's next turn.
            _burning_mask = self.fire_map == BURNING
            _fire_adj = binary_dilation(_burning_mask, structure=_CROSS_KERNEL)
            for (_cr, _cc), _owner in list(self.mitigation_owner.items()):
                _mit_status = self.fire_map[_cr, _cc]
                if (
                    _mit_status in _MIT_CONTACT_BONUS_INDIV
                    and _fire_adj[_cr, _cc]
                ):
                    self.pending_agent_rewards[_owner] += (
                        _MIT_CONTACT_BONUS_INDIV[_mit_status]
                    )

            # ---- Per-station growth penalty (individual, not shared) --------------
            # For every cell that newly ignited this sub-step, find the fire station
            # with the smallest Manhattan distance to that cell (Voronoi ownership).
            # All agents from that station receive a −50 penalty for each cell that
            # burned in their territory — fire spreading in your zone is your fault.
            # Agents from other stations are NOT penalised for those cells.
            _burned_now = (self.fire_map == BURNED) | (self.fire_map == BURNING)
            _newly_burned_cells = np.argwhere(_burned_now & ~self.prev_burned_mask)
            if len(_newly_burned_cells) > 0 and len(self.station_positions) > 0:
                # Pairwise Manhattan distance: (n_cells, n_stations)
                _dists_cells_to_stations = (
                    np.abs(_newly_burned_cells[:, 0:1] - self.station_positions[:, 0])
                    + np.abs(_newly_burned_cells[:, 1:2] - self.station_positions[:, 1])
                )
                _closest_station = np.argmin(_dists_cells_to_stations, axis=1)  # (n,)
                # Count how many cells burned in each station's Voronoi sector
                for _s_idx in range(len(self.station_positions)):
                    _n_sector_cells = int(np.sum(_closest_station == _s_idx))
                    if _n_sector_cells > 0:
                        _sector_penalty = _GROWTH_PENALTY_PER_CELL * _n_sector_cells
                        for _ai, _ag in enumerate(self.agents):
                            if _ag.get("station_idx") == _s_idx:
                                self.pending_agent_rewards[_ai] -= _sector_penalty

            # Advance sub-step counter
            self.fire_sub_step += 1
            batches = self.fire_schedule.get(self.fire_timestep, [])
            if self.fire_sub_step >= max(len(batches), self.actions_per_fire_step):
                # ---- Individual blocked-cells reward (at fire timestep boundary) --
                # Cells with fire_arrival == fire_timestep that still carry mitigation
                # were SCHEDULED to burn but the placing agent's barrier stopped them.
                # Credit goes to the placing agent via pending_agent_rewards.
                _should_burn = arr_full == self.fire_timestep
                for (_cr, _cc), _owner in list(self.mitigation_owner.items()):
                    if _should_burn[_cr, _cc] and self.fire_map[_cr, _cc] in (
                        FIRELINE, SCRATCHLINE, WETLINE
                    ):
                        self.pending_agent_rewards[_owner] += _BLOCKED_CELLS_REWARD

                self._transition_burning_to_burned()
                self.fire_timestep += 1
                self.fire_sub_step = 0
                for a in self.agents:
                    a["time_remaining"] = self.time_budget
                if self.fire_timestep > self.max_fire_timestep:
                    self._is_active = False

        # ---- Flat step penalty (−1 per step, uniform) ----
        reward += fighting_fire_step_penalty([agent["pos"]], self.fire_map)

        # Apply reward scale (keeps PPO value function stable)
        reward *= self.reward_scale

        self.agent_pos = self.agents[0]["pos"]
        obs = self._build_observation()

        self.num_steps += 1
        truncated = self.num_steps >= self.max_episode_steps
        terminated = not self._is_active

        info = {
            "num_steps": self.num_steps,
            "fire_timestep": self.fire_timestep,
            "fire_sub_step": self.fire_sub_step,
            "current_agent": self.current_agent_idx,
            "num_agents": self.num_agents,
            # Time budget diagnostics (useful for debugging / reward shaping)
            "time_remaining": agent["time_remaining"],
            "time_budget": self.time_budget,
            "time_remaining_norm": agent["time_remaining"] / max(1.0, self.time_budget),
            "budget_exceeded": _budget_exceeded,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def build_terrain_base_rgb(self) -> np.ndarray:
        """Return a (grid_rows, grid_cols, 3) uint8 RGB base frame showing:
          - Elevation hillshade (light/shadow)
          - Population density overlay (yellow → red warm tones)
          - Roads overlay (white/grey lines)
        """
        elev = np.load("data/camp_fire_elevation_375m.npy").astype(np.float32)
        elev = elev[: self.grid_rows, : self.grid_cols]
        elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)

        dy, dx = np.gradient(elev)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        aspect = np.arctan2(-dx, dy)
        altitude_rad = np.radians(45)
        azimuth_rad = np.radians(315)
        shade = np.sin(altitude_rad) * np.cos(np.arctan(slope)) + np.cos(
            altitude_rad
        ) * np.sin(np.arctan(slope)) * np.cos(azimuth_rad - aspect)
        shade = np.clip(shade, 0.15, 1.0)

        frame = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.float32)
        frame[:, :, 0] = (0.35 + 0.50 * elev_norm) * shade
        frame[:, :, 1] = (0.55 - 0.25 * elev_norm) * shade
        frame[:, :, 2] = (0.25 + 0.20 * elev_norm) * shade

        pop = self.population_grid[: self.grid_rows, : self.grid_cols]
        pop_max = float(pop.max()) if pop.max() > 0 else 1.0
        pop_mask = pop > 0
        if np.any(pop_mask):
            p = np.where(pop_mask, pop / pop_max, 0.0)
            alpha_p = np.where(
                pop_mask, np.minimum(0.85, 0.55 + 0.30 * p), 0.0
            )
            pr = np.minimum(1.0, 1.00)
            pg = np.maximum(0.0, 0.80 - 0.80 * p)
            pb = np.where(pop_mask, 0.05, 0.0)
            for ch, layer in enumerate([pr, pg, pb]):
                frame[:, :, ch] = np.where(
                    pop_mask,
                    frame[:, :, ch] * (1.0 - alpha_p) + layer * alpha_p,
                    frame[:, :, ch],
                )

        road_colors_f = {
            1: (1.00, 1.00, 1.00),
            2: (0.78, 0.78, 0.78),
            3: (0.65, 0.65, 0.65),
            4: (0.55, 0.55, 0.45),
        }
        roads = self.roads_grid[: self.grid_rows, : self.grid_cols]
        road_alpha = 0.70
        for rt, (rr, rg, rb) in road_colors_f.items():
            mask = roads == rt
            if np.any(mask):
                for ch, lc in enumerate([rr, rg, rb]):
                    frame[:, :, ch] = np.where(
                        mask,
                        frame[:, :, ch] * (1.0 - road_alpha) + lc * road_alpha,
                        frame[:, :, ch],
                    )

        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    def _get_sector_target(
        self, agent_idx: int, r: int, c: int
    ) -> Optional[Tuple[int, int]]:
        """Return the Voronoi sector intercept target for agent_idx.

        Algorithm
        ---------
        1. Dynamic k:  k = max(1, min(4, ceil(dist_to_nearest_burning /
                                             actions_per_fire_step)))
           A far agent → k=3/4 → targets a future front; intercepts rather
           than chases.  A near agent → k=1 → targets the imminent front.

        2. Find all cells with fire_arrival == fire_timestep + k.
           Fall back to k+1, k-1, then current BURNING cells if empty.

        3. Voronoi partition by STATION proximity:
           For each candidate cell, compute the Manhattan distance to every
           fire station.  The cell belongs to the station whose distance is
           smallest (ties broken by station index).  Agent i's sector is all
           candidate cells that belong to agent i's home station.

           This naturally assigns the NW portion of the fire front to the NW
           station, the SE portion to the SE station, etc. — stations near the
           fire edge they can realistically cover.

        4. Return the centroid of agent i's sector cells.

        If agent i's station has no cells in its sector (e.g., all fire cells
        are closer to another station), fall back to the NEAREST candidate cell
        to agent i's station position.

        Multiple trucks from the same station share the same target centroid;
        their slightly different starting positions and the ID channel break
        symmetry enough for the policy to assign them different sub-roles.
        """
        arr_full = self.fire_arrival[: self.grid_rows, : self.grid_cols]
        station_idx = self.agents[agent_idx].get("station_idx", 0)
        s_pos = self.station_positions  # (num_stations, 2)

        # ---- Step 1: dynamic k ----
        burning_rs, burning_cs = np.where(
            self.fire_map[: self.grid_rows, : self.grid_cols] == BURNING
        )
        if len(burning_rs) > 0:
            min_dist = int(
                np.min(np.abs(burning_rs - r) + np.abs(burning_cs - c))
            )
            k = max(1, min(4, math.ceil(min_dist / max(1, self.actions_per_fire_step))))
        else:
            k = 1

        # ---- Step 2: candidate cells ----
        target_cells: Optional[np.ndarray] = None
        for try_k in [k, min(k + 1, 4), max(k - 1, 1)]:
            candidates = np.argwhere(arr_full == self.fire_timestep + try_k)
            if len(candidates) > 0:
                target_cells = candidates
                break

        if target_cells is None or len(target_cells) == 0:
            # Ultimate fallback: nearest burning cell to agent's station
            if len(burning_rs) > 0:
                sr, sc = int(s_pos[station_idx, 0]), int(s_pos[station_idx, 1])
                dists = np.abs(burning_rs - sr) + np.abs(burning_cs - sc)
                idx = int(dists.argmin())
                return (int(burning_rs[idx]), int(burning_cs[idx]))
            return None

        # ---- Step 3: Voronoi partition by station proximity ----
        # target_cells: (n, 2);  s_pos: (m, 2)
        # dists[i, j] = manhattan dist from candidate cell i to station j
        dists_to_stations = (
            np.abs(target_cells[:, 0:1] - s_pos[:, 0])   # (n, m) row diffs
            + np.abs(target_cells[:, 1:2] - s_pos[:, 1]) # (n, m) col diffs
        )  # shape (n, m)
        closest_station = np.argmin(dists_to_stations, axis=1)  # (n,) int

        # ---- Step 4: agent's sector = cells closest to their station ----
        my_mask = closest_station == station_idx
        if np.any(my_mask):
            sector = target_cells[my_mask]
        else:
            # Station has no Voronoi cells — fall back to nearest candidate
            # to the station itself (not the agent, so trucks stay on-target)
            sr, sc = int(s_pos[station_idx, 0]), int(s_pos[station_idx, 1])
            fallback_dists = (
                np.abs(target_cells[:, 0] - sr)
                + np.abs(target_cells[:, 1] - sc)
            )
            sector = target_cells[np.argmin(fallback_dists) : np.argmin(fallback_dists) + 1]

        return (int(np.mean(sector[:, 0])), int(np.mean(sector[:, 1])))

    def _build_observation(self, fire_map_override=None) -> np.ndarray:
        """Build a full-grid centred observation window for the current agent.

        Raw window: (obs_window_size × obs_window_size) centred on the agent.
        The window is larger than the grid diagonal, so the entire Camp Fire
        extent is always visible regardless of the agent's position.
        Cells outside the grid are zero-padded.

        After filling 8 channels, the window is stride-downsampled by
        obs_downsample → shape (obs_out_size, obs_out_size, 8).

        8 channels:
          0  fire_map      — normalised burn/mitigation status  [0, 1]
          1  fire_ahead_1  — binary: fire_arrival == fire_timestep+1
          2  fire_ahead_2  — binary: fire_arrival == fire_timestep+2
          3  fire_ahead_3  — binary: fire_arrival == fire_timestep+3
          4  fire_ahead_4  — binary: fire_arrival == fire_timestep+4
          5  other_agents  — binary: 1.0 at every other agent's cell
          6  mitigation    — binary: 1.0 where any mitigation exists
          7  agent_id_norm — constant = agent_idx / (num_agents−1)
                             Breaks observation symmetry so that even when
                             agents are co-located they see different inputs
                             and the shared policy can assign different roles.
        """
        win = self.obs_window_size
        half = win // 2
        ag = self.agents[self.current_agent_idx]
        ag_r, ag_c = ag["pos"]

        window = np.zeros((win, win, OBS_N_CHANNELS), dtype=np.float32)

        # Grid slice (clamped to valid grid range)
        g_r0 = max(0, ag_r - half)
        g_r1 = min(self.grid_rows, ag_r + half + 1)
        g_c0 = max(0, ag_c - half)
        g_c1 = min(self.grid_cols, ag_c + half + 1)

        # Corresponding slice inside the raw window (handles edge zero-padding)
        w_r0 = half - (ag_r - g_r0)
        w_r1 = w_r0 + (g_r1 - g_r0)
        w_c0 = half - (ag_c - g_c0)
        w_c1 = w_c0 + (g_c1 - g_c0)

        fm_slice = self.fire_map[g_r0:g_r1, g_c0:g_c1]
        arr_slice = self.fire_arrival[g_r0:g_r1, g_c0:g_c1]

        # Channel 0: fire_map normalised (WETLINE=5 → 1.0)
        window[w_r0:w_r1, w_c0:w_c1, 0] = (
            fm_slice.astype(np.float32) / float(WETLINE)
        )

        # Channels 1–4: binary fire prediction layers
        for i, t_off in enumerate(range(1, 5)):
            window[w_r0:w_r1, w_c0:w_c1, 1 + i] = (
                arr_slice == self.fire_timestep + t_off
            ).astype(np.float32)

        # Channel 5: other agents (binary)
        for i, other in enumerate(self.agents):
            if i != self.current_agent_idx:
                or_, oc = other["pos"]
                if g_r0 <= or_ < g_r1 and g_c0 <= oc < g_c1:
                    window[w_r0 + (or_ - g_r0), w_c0 + (oc - g_c0), 5] = 1.0

        # Channel 6: existing mitigation (binary)
        mit_mask = (
            (fm_slice == FIRELINE)
            | (fm_slice == SCRATCHLINE)
            | (fm_slice == WETLINE)
        )
        window[w_r0:w_r1, w_c0:w_c1, 6] = mit_mask.astype(np.float32)

        # Channel 7: agent-ID normalised to [0, 1]
        # Constant across all spatial positions — unique per agent.
        # Even when two agents share the same cell (same channels 0–6),
        # their channel-7 values differ, so the policy can output different
        # actions for each (role specialisation without explicit coordination).
        id_norm = (
            self.current_agent_idx / max(1, self.num_agents - 1)
        )
        window[:, :, 7] = id_norm

        # Stride-downsample: [::d, ::d] is O(1) view, no copy needed
        if self.obs_downsample > 1:
            window = window[::self.obs_downsample, ::self.obs_downsample, :]

        return window.astype(np.float32)

    def _ignite_full_timestep(self, t: int) -> None:
        """Ignite all cells for a given fire timestep at once (t=0 only)."""
        newly_burning = (self.fire_arrival == t) & (self.fire_map == UNBURNED)
        self.fire_map[newly_burning] = BURNING

    def _is_fire_adjacent(self, r: int, c: int) -> bool:
        """Return True if (r, c) is 4-connected to any BURNING or BURNED cell."""
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                if self.fire_map[nr, nc] in (BURNING, BURNED):
                    return True
        return False

    def _advance_fire_substep(self) -> None:
        """Ignite the next batch of cells for the current fire timestep.

        A cell ignites only if it is UNBURNED and 4-adjacent to fire.
        Mitigation cells are permanent barriers (fireline) or degrade after
        repeated fire-adjacency (scratchline / wetline).
        """
        batches = self.fire_schedule.get(self.fire_timestep, [])
        if self.fire_sub_step < len(batches):
            for rr, cc in batches[self.fire_sub_step]:
                if (
                    self.fire_map[rr, cc] == UNBURNED
                    and self._is_fire_adjacent(rr, cc)
                ):
                    self.fire_map[rr, cc] = BURNING

        # Mitigation degradation (scratchline / wetline)
        newly_unburned: List[Tuple[int, int]] = []
        for mit_type, threshold in _MITIGATION_RESISTANCE.items():
            mit_rows, mit_cols = np.where(self.fire_map == mit_type)
            for rr, cc in zip(mit_rows.tolist(), mit_cols.tolist()):
                if self._is_fire_adjacent(rr, cc):
                    key = (rr, cc)
                    self.mitigation_exposure[key] = (
                        self.mitigation_exposure.get(key, 0) + 1
                    )
                    if self.mitigation_exposure[key] >= threshold:
                        self.fire_map[rr, cc] = UNBURNED
                        del self.mitigation_exposure[key]
                        newly_unburned.append(key)

        for rr, cc in newly_unburned:
            # Degraded mitigation is gone — remove ownership record
            self.mitigation_owner.pop((rr, cc), None)
            if self._is_fire_adjacent(rr, cc):
                self.fire_map[rr, cc] = BURNING

    def _transition_burning_to_burned(self) -> None:
        """Transition all BURNING cells to BURNED at end of a fire timestep."""
        self.fire_map[self.fire_map == BURNING] = BURNED
