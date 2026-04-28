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

Reward (per agent step):
  1. Approach reward: +3 raw per cell closer to the nearest fire_ahead_1
     cell within the search radius (falls back to nearest BURNING cell).
     Targeting the PREDICTED fire front (not current burning) teaches
     agents to intercept before the fire arrives.
  2. Placement rewards (only when placing mitigation on UNBURNED cell):
     a. Near-fire bonus: +50 raw if within _PLACEMENT_REWARD_RADIUS cells
        of a BURNING cell or a fire_ahead_1/2 cell.
     b. Chain bonus: +_CHAIN_BONUS additional reward when the new cell is
        4-adjacent to an EXISTING mitigation cell AND within
        _CHAIN_FIRE_RADIUS of active fire.  Teaches line-building instead
        of isolated dot placement — an unbroken barrier actually stops fire.
     c. Wasted penalty: −10 raw when neither near fire nor chain-connected.
  3. Blocked-cells reward (once per fire timestep, at transition):
     +_BLOCKED_CELLS_REWARD raw per cell where fire_arrival==fire_timestep
     AND fire_map has mitigation.  PRIMARY effectiveness signal: directly
     measures whether agents' barriers actually stopped cells that would
     otherwise have burned.
  4. Round reward (once per agent round, after fire sub-step advances):
     - Mitigation-contact bonus (from fighting_fire_round_reward)
     - Fire-front growth penalty: −50 × new cells ignited
     - Population penalty: −100 × people in newly burned cells
  5. Step penalty (per agent, per step): 0 / −0.5 / −2.0 based on
     proximity to active fire (from fighting_fire_step_penalty).
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from scipy.ndimage import distance_transform_edt

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

# Placement reward: radius for "near fire" check (cells)
_PLACEMENT_REWARD_RADIUS: int = 5

# Near-fire placement bonus: agent placed mitigation within proximity of fire.
_NEAR_FIRE_PLACEMENT_BONUS: float = 50.0

# Wasted placement penalty: discourage random mitigation far from fire.
_WASTED_PLACEMENT_PENALTY: float = 10.0

# Chain bonus: reward for extending an existing mitigation line.
# Awarded when the newly placed cell is 4-adjacent to existing mitigation
# AND within _CHAIN_FIRE_RADIUS cells of active BURNING fire.
# A single isolated cell does nothing — fire flows around it.  An
# unbroken line of mitigation cells forms a true barrier.
_CHAIN_BONUS: float = 150.0
_CHAIN_FIRE_RADIUS: int = 15

# Blocked-cells reward (raw per cell): paid at each fire timestep transition
# for every cell that had fire_arrival==fire_timestep AND has mitigation.
# This is the only reward signal that measures ACTUAL fire-stopping
# effectiveness rather than just placement proximity.
_BLOCKED_CELLS_REWARD: float = 300.0

# Approach reward: dense directional signal toward the predicted fire front.
# Set to cover the whole grid (half of 220 cells after downsampling back to
# grid coords = 110 cells) so agents always get a gradient toward distant fire.
_APPROACH_SEARCH_RADIUS: int = 110  # cells to search for fire_ahead_1 target
_APPROACH_REWARD_PER_CELL: float = 3.0  # raw reward per cell of approach

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

        self.agents = []
        occupied: set = set()
        for station in self.stations:
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

        # ---- Move agent ----
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
        if new_pos != list(agent["pos"]):
            agent["pos"] = new_pos

        reward = 0.0
        r, c = agent["pos"]
        _old_r, _old_c = _old_pos[0], _old_pos[1]

        # Pre-slice fire arrival grid to grid bounds (reused multiple times below)
        arr_full = self.fire_arrival[: self.grid_rows, : self.grid_cols]

        # ---- Approach reward: move toward the PREDICTED fire front ----
        # Primary target: cells burning at fire_timestep+1 (3 hours ahead).
        # Fallback: nearest currently BURNING cell (late episode when no
        # fire_ahead_1 cells remain in the search window).
        # Targeting the future front — not the current burning edge — teaches
        # agents to intercept and cut off fire before it arrives.
        if [r, c] != [_old_r, _old_c]:
            _sr0 = max(0, r - _APPROACH_SEARCH_RADIUS)
            _sr1 = min(self.grid_rows, r + _APPROACH_SEARCH_RADIUS + 1)
            _sc0 = max(0, c - _APPROACH_SEARCH_RADIUS)
            _sc1 = min(self.grid_cols, c + _APPROACH_SEARCH_RADIUS + 1)

            _ahead_sub = arr_full[_sr0:_sr1, _sc0:_sc1]
            _frs, _fcs = np.where(_ahead_sub == self.fire_timestep + 1)

            if len(_frs) == 0:
                # Fallback: current burning cells
                _burn_sub = self.fire_map[_sr0:_sr1, _sc0:_sc1]
                _frs, _fcs = np.where(_burn_sub == BURNING)

            if len(_frs) > 0:
                _frs = _frs + _sr0
                _fcs = _fcs + _sc0
                _new_dist = float(
                    np.min(np.abs(_frs - r) + np.abs(_fcs - c))
                )
                _old_dist = float(
                    np.min(np.abs(_frs - _old_r) + np.abs(_fcs - _old_c))
                )
                _delta = _old_dist - _new_dist  # positive = moved closer
                if _delta > 0:
                    reward += _delta * _APPROACH_REWARD_PER_CELL

        # ---- Placement reward ----
        # Rewards only apply when placing mitigation on an UNBURNED cell.
        # Two-tier signal:
        #   a. Near-fire bonus (+50): placement is within _PLACEMENT_REWARD_RADIUS
        #      of a BURNING cell or a fire_ahead_1/2 cell.
        #   b. Chain bonus (+150): new cell is 4-adjacent to existing mitigation
        #      AND within _CHAIN_FIRE_RADIUS of active fire — teaches line-building.
        #   c. Wasted penalty (-10): placement is far from fire (discourages random
        #      exploratory placements that waste the agent's step).
        cell_value = self.fire_map[r, c]
        if cell_value == UNBURNED and interaction_str != "none":
            mitigation_type = MITIGATION_MAP.get(interaction_str, FIRELINE)
            self.fire_map[r, c] = mitigation_type

            # Check proximity to active/predicted fire
            _pr0 = max(0, r - _PLACEMENT_REWARD_RADIUS)
            _pr1 = min(self.grid_rows, r + _PLACEMENT_REWARD_RADIUS + 1)
            _pc0 = max(0, c - _PLACEMENT_REWARD_RADIUS)
            _pc1 = min(self.grid_cols, c + _PLACEMENT_REWARD_RADIUS + 1)
            _loc_fire = self.fire_map[_pr0:_pr1, _pc0:_pc1]
            _loc_arr = arr_full[_pr0:_pr1, _pc0:_pc1]
            _near_fire = (
                np.any(_loc_fire == BURNING)
                or np.any(_loc_arr == self.fire_timestep + 1)
                or np.any(_loc_arr == self.fire_timestep + 2)
            )

            if _near_fire:
                reward += _NEAR_FIRE_PLACEMENT_BONUS

                # Chain bonus: extending an existing mitigation line.
                _is_chain = False
                for _dr, _dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    _nr, _nc = r + _dr, c + _dc
                    if 0 <= _nr < self.grid_rows and 0 <= _nc < self.grid_cols:
                        if self.fire_map[_nr, _nc] in (FIRELINE, SCRATCHLINE, WETLINE):
                            _is_chain = True
                            break

                if _is_chain:
                    # Verify there is active BURNING fire within chain radius
                    _cr0 = max(0, r - _CHAIN_FIRE_RADIUS)
                    _cr1 = min(self.grid_rows, r + _CHAIN_FIRE_RADIUS + 1)
                    _cc0 = max(0, c - _CHAIN_FIRE_RADIUS)
                    _cc1 = min(self.grid_cols, c + _CHAIN_FIRE_RADIUS + 1)
                    if np.any(self.fire_map[_cr0:_cr1, _cc0:_cc1] == BURNING):
                        reward += _CHAIN_BONUS
            else:
                reward -= _WASTED_PLACEMENT_PENALTY

        # ---- Advance to next agent ----
        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents

        # ---- When all agents have acted: advance fire, compute round reward ----
        if self.current_agent_idx == 0:
            # Snapshot burned state BEFORE fire advances (for delta signals)
            self.prev_burned_mask = (self.fire_map == BURNED) | (
                self.fire_map == BURNING
            )

            # Advance fire by one sub-step
            self._advance_fire_substep()

            # Round reward: mitigation-contact + growth penalty + population penalty
            reward += fighting_fire_round_reward(
                self.fire_map,
                self.population_grid,
                self.prev_burned_mask,
            )

            # Advance sub-step counter
            self.fire_sub_step += 1
            batches = self.fire_schedule.get(self.fire_timestep, [])
            if self.fire_sub_step >= max(len(batches), self.actions_per_fire_step):
                # ---- Blocked-cells reward (at fire timestep transition) ----
                # Cells with fire_arrival == fire_timestep that currently have
                # mitigation were SCHEDULED to burn this timestep but the agents'
                # barriers prevented it.  This is the primary signal that barriers
                # actually worked — not just that they were placed near fire.
                _should_burn = arr_full == self.fire_timestep
                _is_mitigated = (
                    (self.fire_map == FIRELINE)
                    | (self.fire_map == SCRATCHLINE)
                    | (self.fire_map == WETLINE)
                )
                _n_blocked = int(np.count_nonzero(_should_burn & _is_mitigated))
                if _n_blocked > 0:
                    reward += _n_blocked * _BLOCKED_CELLS_REWARD

                self._transition_burning_to_burned()
                self.fire_timestep += 1
                self.fire_sub_step = 0
                for a in self.agents:
                    a["time_remaining"] = self.time_budget
                if self.fire_timestep > self.max_fire_timestep:
                    self._is_active = False

        # ---- Per-agent step penalty: 3-tier based on proximity to fire ----
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
            if self._is_fire_adjacent(rr, cc):
                self.fire_map[rr, cc] = BURNING

    def _transition_burning_to_burned(self) -> None:
        """Transition all BURNING cells to BURNED at end of a fire timestep."""
        self.fire_map[self.fire_map == BURNING] = BURNED
