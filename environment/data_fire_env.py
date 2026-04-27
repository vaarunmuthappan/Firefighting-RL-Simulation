"""Data-driven firefighting RL environment using real Camp Fire VIIRS data.

Fire spread follows the precomputed fire_arrival grid derived from VIIRS
satellite observations (FEDS dataset). Fire advances proportionally — cells
are distributed evenly across sub-steps within each fire timestep to avoid
visual/temporal jumps.

Multi-agent support: multiple fire engines start from real fire station
locations. Each agent has a time budget per fire timestep (180 min for
3-hour steps). Movement on roads is fast (5 min/cell), off-road is slow
(15 min/cell), and building mitigation consumes significant time.

Reward:
  - Area penalty: -proportion of total cells burned/burning
  - Population penalty: -1,000,000 × population of newly burned pixels
  - Timestep penalty: -1000 per agent step
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
from environment.observation_builder import ObservationBuilder
from fire_sim.sim_interface import FireSimInterface
from reward.reward_functions import fighting_fire_round_reward, fighting_fire_step_penalty

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

# Time costs in minutes per action
MOVE_TIME_ROAD = 5.0       # 375m at ~45 km/hr
MOVE_TIME_OFFROAD = 15.0   # 375m at ~25 km/hr effective
BUILD_TIME: Dict[str, float] = {
    "fireline": 45.0,      # 375m at bulldozer rate ~500 m/hr
    "scratchline": 30.0,   # lighter construction
    "wetline": 20.0,       # spray-based, fastest
}

# Mitigation resistance: number of fire-adjacent sub-steps before degrading.
# Each sub-step ≈ 5 minutes (180 min / 36 sub-steps per fire timestep).
# FIRELINE is permanent (not listed here = no degradation).
_MITIGATION_RESISTANCE: Dict[int, int] = {
    SCRATCHLINE: 108,  # ~9 h of adjacency (3 full fire timesteps)
    WETLINE: 36,       # ~3 h of adjacency (1 full fire timestep)
}

# Spiral offsets used to spread trucks from the same station to unique cells.
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
    grid. Each fire timestep (3 hours), newly burning cells are distributed
    proportionally across agent sub-steps so fire expands gradually.

    Multiple agents (fire engines) start at real fire station locations. Each
    agent has a 180-minute time budget per fire timestep. Actions consume time
    based on terrain (road vs off-road) and action type (move vs build).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.screen_size: int = config.get("screen_size", 250)

        # Load fire arrival data
        data_dir = Path(config.get("data_dir", "data"))
        arrival_path = data_dir / config.get("fire_arrival_file", "camp_fire_full_arrival_375m.npy")
        meta_path = data_dir / config.get("fire_meta_file", "camp_fire_full_meta.json")

        self.fire_arrival = np.load(arrival_path)
        with open(meta_path) as f:
            self.fire_meta = json.load(f)

        grid_rows = self.fire_meta.get("grid_rows", self.fire_arrival.shape[0])
        grid_cols = self.fire_meta.get("grid_cols", self.fire_arrival.shape[1])
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        assert self.fire_arrival.shape == (grid_rows, grid_cols), \
            f"fire_arrival shape {self.fire_arrival.shape} != ({grid_rows}, {grid_cols})"

        self.max_fire_timestep = int(self.fire_arrival.max())
        self.hours_per_timestep = self.fire_meta.get("hours_per_timestep", 3.0)
        self.time_budget = self.hours_per_timestep * 60.0  # minutes per fire timestep

        # Load roads grid
        roads_path = data_dir / config.get("roads_file", "camp_fire_roads_375m.npy")
        if roads_path.exists():
            self.roads_grid = np.load(roads_path)
        else:
            self.roads_grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)

        # Load population grid
        pop_path = data_dir / config.get("population_file", "camp_fire_population_375m.npy")
        if pop_path.exists():
            self.population_grid = np.load(pop_path).astype(np.float32)
        else:
            self.population_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        # Load fire stations
        stations_path = data_dir / config.get("stations_file", "camp_fire_stations.json")
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

        # Config
        self.attributes: List[str] = list(config.get("attributes", [
            "fire_map", "elevation", "w_0", "sigma", "delta", "M_x"
        ]))
        _norm_attrs: List[str] = list(config.get("normalized_attributes", [
            "elevation", "w_0", "sigma", "delta", "M_x"
        ]))
        if "fire_map" not in _norm_attrs:
            _norm_attrs = ["fire_map"] + _norm_attrs
        self.normalized_attributes = _norm_attrs

        self.initial_pos: List[int] = list(config.get("initial_pos", [15, 15]))
        self.max_episode_steps: int = config.get("max_episode_steps", 5000)

        movements: List[str] = list(config.get("movements", ["up", "down", "left", "right"]))
        interactions: List[str] = list(config.get("interactions", ["fireline"]))
        self.movements: List[str] = ["none"] + movements
        self.interactions: List[str] = ["none"] + interactions

        max_mitigation = max(
            (MITIGATION_MAP.get(name, FIRELINE) for name in interactions),
            default=FIRELINE,
        )
        self.sim_agent_id: int = max_mitigation + 2

        # Count total agents (trucks) across all in-grid stations
        self.num_agents = sum(s.get("trucks", 1) for s in self.stations)
        if self.num_agents == 0:
            self.num_agents = 1  # fallback single agent

        # Actions per agent step: each agent picks (movement, interaction)
        self.actions_per_agent = len(self.movements) * len(self.interactions)

        # Pre-compute proportional fire spread schedule
        self._precompute_fire_schedule(config)

        # Build SimFire for terrain observations only
        sim_config = FireSimInterface.build_config_dict(
            screen_size=max(grid_rows, grid_cols),
            terrain_type=config.get("terrain_type", "operational"),
            latitude=config.get("latitude", self.fire_meta.get("lat_min", 39.58)),
            longitude=config.get("longitude", self.fire_meta.get("lon_min", -121.79)),
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

        # Normalisation bounds
        # self_pos / other_agents are binary [0,1] — handled by the else branch.
        # fire_map uses max=WETLINE=5 (agents are no longer stamped onto fire_map;
        # they get their own dedicated channels instead).
        raw_bounds = dict(self.sim.get_attribute_bounds())
        self.min_maxes: Dict[str, Dict[str, float]] = {}
        for attr in self.attributes:
            if attr == "fire_map":
                self.min_maxes[attr] = {"min": 0, "max": float(WETLINE)}
            elif attr in raw_bounds:
                self.min_maxes[attr] = {
                    "min": float(raw_bounds[attr]["min"]),
                    "max": float(raw_bounds[attr]["max"]),
                }
            else:
                # self_pos, other_agents, and any future binary channels
                self.min_maxes[attr] = {"min": 0.0, "max": 1.0}

        # Observation downsampling for memory efficiency
        # Default 4x downsample: (101,107,6) → (26,27,6) = 4,212 floats
        # vs full (101,107,6) = 64,842 floats. Buffer of 10k: ~337MB vs ~25GB.
        self.obs_downsample = config.get("obs_downsample", 4)
        # Use ceiling division to match actual slice [::obs_downsample] output size
        import math
        self.obs_rows = max(1, math.ceil(grid_rows / self.obs_downsample))
        self.obs_cols = max(1, math.ceil(grid_cols / self.obs_downsample))

        # Spaces — sequential multi-agent: one action per step for current agent
        num_attrs = len(self.attributes)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_rows, self.obs_cols, num_attrs),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.actions_per_agent)

        # Observation builder
        self.obs_builder = ObservationBuilder(
            sim_interface=self.sim,
            attributes=self.attributes,
            normalized_attributes=self.normalized_attributes,
            min_maxes=self.min_maxes,
        )

        self.spec = EnvSpec(
            id="DataFireEnv-v0",
            entry_point="environment.data_fire_env:DataDrivenFireEnv",
            max_episode_steps=self.max_episode_steps,
        )

        # State (set properly in reset)
        self.agents: List[Dict[str, Any]] = []
        self.current_agent_idx: int = 0
        self.num_steps: int = 0
        self.fire_timestep: int = 0
        self.fire_sub_step: int = 0  # sub-step within current fire timestep
        self.fire_map: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=int)
        self.prev_burned_mask: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=bool)
        self._is_active: bool = True
        # Tracks how many fire-adjacent sub-steps each degradable mitigation cell has endured
        self.mitigation_exposure: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_fire_schedule(self, config: dict) -> None:
        """Pre-compute which cells ignite at each sub-step for proportional spread.

        For each fire timestep t, cells with arrival==t are sorted by distance
        from the t-1 fire perimeter and distributed evenly across sub-steps.

        Optimized: build cumulative fire mask incrementally instead of calling
        distance_transform_edt on the full grid each timestep.
        """
        self.actions_per_fire_step = config.get("actions_per_fire_step", 36)

        self.fire_schedule: Dict[int, List[List[Tuple[int, int]]]] = {}

        # Build sorted cell lists incrementally
        # cumulative_fire tracks which cells have arrived up to t-1
        cumulative_fire = np.zeros((self.fire_arrival.shape[0], self.fire_arrival.shape[1]), dtype=bool)

        for t in range(0, self.max_fire_timestep + 1):
            cell_coords = np.argwhere(self.fire_arrival == t)
            if len(cell_coords) == 0:
                # Update cumulative (nothing new)
                self.fire_schedule[t] = []
                continue

            if t == 0 or not np.any(cumulative_fire):
                # First timestep or no prior fire: all cells ignite at sub-step 0
                self.fire_schedule[t] = [list(map(tuple, cell_coords))]
            else:
                # Sort cells by distance from existing fire front using EDT
                # Only compute EDT once per timestep (not per sub-step)
                dist = distance_transform_edt(~cumulative_fire)
                distances = dist[cell_coords[:, 0], cell_coords[:, 1]]
                order = np.argsort(distances)
                cell_coords = cell_coords[order]

                # Distribute across sub-steps
                n_sub = self.actions_per_fire_step
                batches: List[List[Tuple[int, int]]] = []
                if len(cell_coords) <= n_sub:
                    for cell in cell_coords:
                        batches.append([(int(cell[0]), int(cell[1]))])
                else:
                    splits = np.array_split(cell_coords, n_sub)
                    for split in splits:
                        batches.append([(int(r), int(c)) for r, c in split])

                self.fire_schedule[t] = batches

            # Update cumulative mask for next timestep
            cumulative_fire[cell_coords[:, 0], cell_coords[:, 1]] = True

    def _create_entry_stations(self, all_stations: list) -> list:
        """Create virtual entry stations from nearest out-of-grid stations."""
        center_lat = (self.fire_meta["lat_min"] + self.fire_meta["lat_max"]) / 2
        center_lon = (self.fire_meta["lon_min"] + self.fire_meta["lon_max"]) / 2

        out_of_grid = [s for s in all_stations if not s.get("in_grid", False)]
        if not out_of_grid:
            return [{"name": "default", "grid_row": 0, "grid_col": 0, "trucks": 1, "in_grid": True}]

        # Sort by distance to grid center
        for s in out_of_grid:
            s["_dist"] = ((s["lat"] - center_lat) ** 2 + (s["lon"] - center_lon) ** 2) ** 0.5

        out_of_grid.sort(key=lambda s: s["_dist"])
        nearest = out_of_grid[:2]

        entry_stations = []
        for s in nearest:
            # Determine direction from center
            dlat = s["lat"] - center_lat
            dlon = s["lon"] - center_lon

            # Find entry row/col on grid edge
            if abs(dlat) > abs(dlon):
                entry_row = 0 if dlat > 0 else self.grid_rows - 1
                # Map lon to col
                col_frac = (s["lon"] - self.fire_meta["lon_min"]) / (self.fire_meta["lon_max"] - self.fire_meta["lon_min"])
                entry_col = max(0, min(self.grid_cols - 1, int(col_frac * self.grid_cols)))
            else:
                entry_col = 0 if dlon < 0 else self.grid_cols - 1
                row_frac = (self.fire_meta["lat_max"] - s["lat"]) / (self.fire_meta["lat_max"] - self.fire_meta["lat_min"])
                entry_row = max(0, min(self.grid_rows - 1, int(row_frac * self.grid_rows)))

            # Snap to nearest road cell on that edge
            if self.roads_grid is not None:
                entry_row, entry_col = self._snap_to_road(entry_row, entry_col)

            entry_stations.append({
                "name": f"Entry from {s['name']}",
                "grid_row": entry_row,
                "grid_col": entry_col,
                "trucks": s.get("trucks", 1),
                "in_grid": True,
            })

        return entry_stations

    def _snap_to_road(self, row: int, col: int, search_radius: int = 10) -> Tuple[int, int]:
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

        # Reset SimFire (for terrain data)
        self.sim.reset()

        # Reset mitigation degradation tracking
        self.mitigation_exposure = {}

        # Initialize agents at fire station positions.
        # Trucks from the same station are spread to adjacent unique cells using
        # spiral offsets so they don't all stack on the same pixel.
        self.agents = []
        occupied: set = set()
        for station in self.stations:
            base_r, base_c = station["grid_row"], station["grid_col"]
            for truck_idx in range(station.get("trucks", 1)):
                pos = None
                for dr, dc in _SPREAD_OFFSETS:
                    r = max(0, min(self.grid_rows - 1, base_r + dr))
                    c = max(0, min(self.grid_cols - 1, base_c + dc))
                    if (r, c) not in occupied:
                        pos = [r, c]
                        occupied.add((r, c))
                        break
                if pos is None:
                    pos = [base_r, base_c]  # fallback (very unlikely)
                self.agents.append({
                    "pos": pos,
                    "station": station.get("name", "unknown"),
                    "truck_id": truck_idx,
                    "time_remaining": self.time_budget,
                })

        # Fallback: if no stations, place single agent at initial_pos
        if not self.agents:
            self.agents = [{
                "pos": copy.copy(self.initial_pos),
                "station": "default",
                "truck_id": 0,
                "time_remaining": self.time_budget,
            }]
            self.num_agents = 1

        self.current_agent_idx = 0
        self.num_steps = 0
        self.fire_timestep = 0
        self.fire_sub_step = 0
        self._is_active = True

        # Build initial fire_map
        self.fire_map = np.full((self.grid_rows, self.grid_cols), UNBURNED, dtype=int)
        self.prev_burned_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

        # Ignite timestep 0 cells
        self._ignite_full_timestep(0)

        # For compatibility, set agent_pos to first agent
        self.agent_pos = self.agents[0]["pos"]

        obs = self._build_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        agent = self.agents[self.current_agent_idx]

        # Decode action
        movement_idx = action % len(self.movements)
        interaction_idx = action // len(self.movements)
        movement_str = self.movements[movement_idx]
        interaction_str = self.interactions[interaction_idx]

        # Compute movement time cost
        move_cost = 0.0
        new_pos = list(agent["pos"])
        if movement_str != "none":
            if movement_str == "up":
                new_pos[0] = max(0, new_pos[0] - 1)
            elif movement_str == "down":
                new_pos[0] = min(self.grid_rows - 1, new_pos[0] + 1)
            elif movement_str == "left":
                new_pos[1] = max(0, new_pos[1] - 1)
            elif movement_str == "right":
                new_pos[1] = min(self.grid_cols - 1, new_pos[1] + 1)

            # Check if actually moved (not clamped at boundary)
            if new_pos != list(agent["pos"]):
                on_road = self.roads_grid[new_pos[0], new_pos[1]] > 0
                move_cost = MOVE_TIME_ROAD if on_road else MOVE_TIME_OFFROAD

        # Compute interaction time cost
        build_cost = 0.0
        if interaction_str != "none":
            build_cost = BUILD_TIME.get(interaction_str, 45.0)

        total_cost = move_cost + build_cost

        # Only execute action if agent has enough time remaining
        if total_cost <= agent["time_remaining"] + 0.01:  # small epsilon for float
            # Move agent
            if move_cost > 0:
                agent["pos"] = new_pos

            # Place mitigation on unburned cells
            cell_value = self.fire_map[agent["pos"][0], agent["pos"][1]]
            if cell_value == UNBURNED and interaction_str != "none":
                mitigation_type = MITIGATION_MAP.get(interaction_str, FIRELINE)
                self.fire_map[agent["pos"][0], agent["pos"][1]] = mitigation_type

            agent["time_remaining"] -= total_cost
        # else: action rejected, agent stays put

        # Advance to next agent
        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents

        # When all agents have had a turn, advance fire and compute round reward
        reward = 0.0
        if self.current_agent_idx == 0:
            # Snapshot burned state BEFORE fire advances (needed for delta signals)
            self.prev_burned_mask = (self.fire_map == BURNED) | (self.fire_map == BURNING)

            # Advance fire by one sub-step
            self._advance_fire_substep()

            # Round reward: mitigation-contact + growth penalty + population penalty
            reward += fighting_fire_round_reward(
                self.fire_map,
                self.population_grid,
                self.prev_burned_mask,
            )

            # Advance fire timestep bookkeeping
            self.fire_sub_step += 1
            batches = self.fire_schedule.get(self.fire_timestep, [])
            if self.fire_sub_step >= max(len(batches), self.actions_per_fire_step):
                self._transition_burning_to_burned()
                self.fire_timestep += 1
                self.fire_sub_step = 0
                for a in self.agents:
                    a["time_remaining"] = self.time_budget
                if self.fire_timestep > self.max_fire_timestep:
                    self._is_active = False

        # Conditional timestep penalty: -1 if near fire, -5 if far away
        agent_positions = [a["pos"] for a in self.agents]
        reward += fighting_fire_step_penalty(agent_positions, self.fire_map)

        # Update agent_pos for compatibility
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

        This is computed once and can be cached; the fire/agent overlay is
        applied on top per-frame in the GIF recorder.
        """
        import numpy as np

        elev = np.load("data/camp_fire_elevation_375m.npy").astype(np.float32)
        elev = elev[:self.grid_rows, :self.grid_cols]
        elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)

        # Hillshade
        dy, dx = np.gradient(elev)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        aspect = np.arctan2(-dx, dy)
        altitude_rad = np.radians(45)
        azimuth_rad = np.radians(315)
        shade = (np.sin(altitude_rad) * np.cos(np.arctan(slope))
                 + np.cos(altitude_rad) * np.sin(np.arctan(slope))
                 * np.cos(azimuth_rad - aspect))
        shade = np.clip(shade, 0.15, 1.0)  # floor at 0.15 to avoid pitch black

        # Base colour: low=green, mid=brown, high=grey — fully vectorised
        frame = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.float32)
        frame[:, :, 0] = (0.35 + 0.50 * elev_norm) * shade
        frame[:, :, 1] = (0.55 - 0.25 * elev_norm) * shade
        frame[:, :, 2] = (0.25 + 0.20 * elev_norm) * shade

        # Population overlay: yellow (low density) → deep red (high density).
        # Alpha is boosted so even small populations are clearly visible.
        # Paradise, CA had ~26k residents; our estimate is ~15k across 673 cells.
        pop = self.population_grid[:self.grid_rows, :self.grid_cols]
        pop_max = float(pop.max()) if pop.max() > 0 else 1.0
        pop_mask = pop > 0
        if np.any(pop_mask):
            p = np.where(pop_mask, pop / pop_max, 0.0)
            # Stronger alpha: base 0.55, max 0.85 — population always visible
            alpha_p = np.where(pop_mask, np.minimum(0.85, 0.55 + 0.30 * p), 0.0)
            pr = np.minimum(1.0, 1.00)          # full red channel always on
            pg = np.maximum(0.0, 0.80 - 0.80 * p)  # yellow → red as density ↑
            pb = np.where(pop_mask, 0.05, 0.0)
            for ch, layer in enumerate([pr, pg, pb]):
                frame[:, :, ch] = np.where(
                    pop_mask,
                    frame[:, :, ch] * (1.0 - alpha_p) + layer * alpha_p,
                    frame[:, :, ch],
                )

        # Roads overlay: white=primary, light-grey=secondary, grey=residential/track
        road_colors_f = {
            1: (1.00, 1.00, 1.00),
            2: (0.78, 0.78, 0.78),
            3: (0.65, 0.65, 0.65),
            4: (0.55, 0.55, 0.45),
        }
        roads = self.roads_grid[:self.grid_rows, :self.grid_cols]
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
        # Fire map — pure burn-status only (agents NOT stamped here so the
        # channel is consistent regardless of which agent is acting).
        obs_fire_map = np.copy(self.fire_map)

        # Get terrain data from sim and crop to grid size
        attr_data = dict(self.sim.get_attribute_data())
        for key in attr_data:
            arr = np.asarray(attr_data[key], dtype=np.float32)
            if arr.shape != (self.grid_rows, self.grid_cols):
                attr_data[key] = arr[:self.grid_rows, :self.grid_cols]

        attr_data["fire_map"] = obs_fire_map

        # self_pos: binary channel — 1.0 only at the CURRENT agent's cell.
        # This is the key fix for independent multi-agent behaviour: each agent
        # now sees a unique observation so the shared policy can condition on
        # position and learn different actions for different locations.
        self_pos = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        cur = self.agents[self.current_agent_idx]
        self_pos[cur["pos"][0], cur["pos"][1]] = 1.0
        attr_data["self_pos"] = self_pos

        # other_agents: binary channel — 1.0 at every OTHER agent's cell.
        # Lets agents learn to spread out and cover different fire sectors.
        other_agents = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            if i != self.current_agent_idx:
                other_agents[agent["pos"][0], agent["pos"][1]] = 1.0
        attr_data["other_agents"] = other_agents

        # Normalise
        for attr in self.normalized_attributes:
            if attr in attr_data and attr in self.min_maxes:
                bounds = self.min_maxes[attr]
                arr = attr_data[attr].astype(np.float32)
                denom = bounds["max"] - bounds["min"]
                if denom == 0:
                    attr_data[attr] = np.zeros_like(arr)
                else:
                    attr_data[attr] = np.clip((arr - bounds["min"]) / denom, 0.0, 1.0)

        channels = []
        for attr in self.attributes:
            channels.append(np.asarray(attr_data[attr], dtype=np.float32))

        full_obs = np.stack(channels, axis=-1).astype(np.float32)

        # Downsample by taking every obs_downsample-th pixel (fastest, no blur)
        if self.obs_downsample > 1:
            full_obs = full_obs[::self.obs_downsample, ::self.obs_downsample, :]

        return full_obs

    def _ignite_full_timestep(self, t: int) -> None:
        """Ignite all cells for a given fire timestep at once (used for t=0 only).

        No adjacency check — these are ignition-origin cells.
        """
        newly_burning = (self.fire_arrival == t) & (self.fire_map == UNBURNED)
        self.fire_map[newly_burning] = BURNING

    def _is_fire_adjacent(self, r: int, c: int) -> bool:
        """Return True if (r, c) is 4-connected to any BURNING or BURNED cell.

        This makes mitigation lines (fireline/scratchline/wetline) impassable
        barriers: fire can only spread to cells that are adjacent to active fire,
        so a solid wall of mitigation cells blocks all propagation beyond it.
        """
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                if self.fire_map[nr, nc] in (BURNING, BURNED):
                    return True
        return False

    def _advance_fire_substep(self) -> None:
        """Ignite the next batch of cells for the current fire timestep.

        A cell ignites only if:
          1. It is UNBURNED (not already on fire, burned, or mitigated), AND
          2. At least one 4-connected neighbour is BURNING or BURNED.

        Fireline is a permanent barrier.  Scratchline and wetline degrade
        after repeated fire-adjacency sub-steps (_MITIGATION_RESISTANCE thresholds)
        — once degraded they become UNBURNED and fire immediately ignites them.
        """
        batches = self.fire_schedule.get(self.fire_timestep, [])
        if self.fire_sub_step < len(batches):
            for r, c in batches[self.fire_sub_step]:
                if self.fire_map[r, c] == UNBURNED and self._is_fire_adjacent(r, c):
                    self.fire_map[r, c] = BURNING

        # --- Mitigation degradation (scratchline / wetline only) ---
        # For each degradable mitigation cell adjacent to fire, increment its
        # exposure counter.  When the threshold is reached the cell reverts to
        # UNBURNED and is immediately ignited if fire is still adjacent.
        newly_unburned: List[Tuple[int, int]] = []
        for mit_type, threshold in _MITIGATION_RESISTANCE.items():
            mit_rows, mit_cols = np.where(self.fire_map == mit_type)
            for r, c in zip(mit_rows.tolist(), mit_cols.tolist()):
                if self._is_fire_adjacent(r, c):
                    key = (r, c)
                    self.mitigation_exposure[key] = self.mitigation_exposure.get(key, 0) + 1
                    if self.mitigation_exposure[key] >= threshold:
                        self.fire_map[r, c] = UNBURNED
                        del self.mitigation_exposure[key]
                        newly_unburned.append(key)

        # Immediately ignite freshly degraded cells (fire is right next door)
        for r, c in newly_unburned:
            if self._is_fire_adjacent(r, c):
                self.fire_map[r, c] = BURNING

    def _transition_burning_to_burned(self) -> None:
        """Transition all BURNING cells to BURNED."""
        was_burning = self.fire_map == BURNING
        self.fire_map[was_burning] = BURNED
