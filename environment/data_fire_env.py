# """Data-driven firefighting RL environment using real Camp Fire VIIRS data.

# Fire spread follows the precomputed fire_arrival grid derived from VIIRS
# satellite observations (FEDS dataset). Fire advances proportionally — cells
# are distributed evenly across sub-steps within each fire timestep to avoid
# visual/temporal jumps.

# Multi-agent support: multiple fire engines start from real fire station
# locations. Each agent has a time budget per fire timestep (180 min for
# 3-hour steps). Movement on roads is fast (5 min/cell), off-road is slow
# (15 min/cell), and building mitigation consumes significant time.

# Reward:
#   - Area penalty: -proportion of total cells burned/burning
#   - Population penalty: -1,000,000 × population of newly burned pixels
#   - Timestep penalty: -1000 per agent step
# """
# import copy
# import json
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from gymnasium import spaces
# from gymnasium.envs.registration import EnvSpec
# from scipy.ndimage import distance_transform_edt

# from environment.base_env import BaseFireEnv
# from environment.observation_builder import ObservationBuilder
# from fire_sim.sim_interface import FireSimInterface
# from reward.reward_functions import data_driven_reward

# # BurnStatus values
# UNBURNED = 0
# BURNING = 1
# BURNED = 2
# FIRELINE = 3
# SCRATCHLINE = 4
# WETLINE = 5

# MITIGATION_MAP: Dict[str, int] = {
#     "fireline": FIRELINE,
#     "scratchline": SCRATCHLINE,
#     "wetline": WETLINE,
# }

# # Time costs in minutes per action
# MOVE_TIME_ROAD = 5.0       # 375m at ~45 km/hr
# MOVE_TIME_OFFROAD = 15.0   # 375m at ~25 km/hr effective
# BUILD_TIME: Dict[str, float] = {
#     "fireline": 45.0,      # 375m at bulldozer rate ~500 m/hr
#     "scratchline": 30.0,   # lighter construction
#     "wetline": 20.0,       # spray-based, fastest
# }


# class DataDrivenFireEnv(BaseFireEnv):
#     """Gymnasium environment using real fire spread data from VIIRS observations.

#     Fire progresses deterministically based on the precomputed fire_arrival
#     grid. Each fire timestep (3 hours), newly burning cells are distributed
#     proportionally across agent sub-steps so fire expands gradually.

#     Multiple agents (fire engines) start at real fire station locations. Each
#     agent has a 180-minute time budget per fire timestep. Actions consume time
#     based on terrain (road vs off-road) and action type (move vs build).
#     """

#     metadata = {"render_modes": []}

#     def __init__(self, config: dict) -> None:
#         super().__init__()

#         self.screen_size: int = config.get("screen_size", 250)

#         # Load fire arrival data
#         data_dir = Path(config.get("data_dir", "data"))
#         arrival_path = data_dir / config.get("fire_arrival_file", "camp_fire_full_arrival_375m.npy")
#         meta_path = data_dir / config.get("fire_meta_file", "camp_fire_full_meta.json")

#         self.fire_arrival = np.load(arrival_path)
#         with open(meta_path) as f:
#             self.fire_meta = json.load(f)

#         grid_rows = self.fire_meta.get("grid_rows", self.fire_arrival.shape[0])
#         grid_cols = self.fire_meta.get("grid_cols", self.fire_arrival.shape[1])
#         self.grid_rows = grid_rows
#         self.grid_cols = grid_cols

#         assert self.fire_arrival.shape == (grid_rows, grid_cols), \
#             f"fire_arrival shape {self.fire_arrival.shape} != ({grid_rows}, {grid_cols})"

#         self.max_fire_timestep = int(self.fire_arrival.max())
#         self.hours_per_timestep = self.fire_meta.get("hours_per_timestep", 3.0)
#         self.time_budget = self.hours_per_timestep * 60.0  # minutes per fire timestep

#         # Load roads grid
#         roads_path = data_dir / config.get("roads_file", "camp_fire_roads_375m.npy")
#         if roads_path.exists():
#             self.roads_grid = np.load(roads_path)
#         else:
#             self.roads_grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)

#         # Load population grid
#         pop_path = data_dir / config.get("population_file", "camp_fire_population_375m.npy")
#         if pop_path.exists():
#             self.population_grid = np.load(pop_path).astype(np.float32)
#         else:
#             self.population_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

#         # Load fire stations
#         stations_path = data_dir / config.get("stations_file", "camp_fire_stations.json")
#         if stations_path.exists():
#             with open(stations_path) as f:
#                 all_stations = json.load(f)
#             self.stations = [s for s in all_stations if s.get("in_grid", False)]
#         else:
#             self.stations = []

#         # If no stations in grid, create entry points from nearest out-of-grid stations
#         if not self.stations and stations_path.exists():
#             with open(stations_path) as f:
#                 all_stations = json.load(f)
#             self.stations = self._create_entry_stations(all_stations)

#         # Config
#         self.attributes: List[str] = list(config.get("attributes", [
#             "fire_map", "elevation", "w_0", "sigma", "delta", "M_x"
#         ]))
#         _norm_attrs: List[str] = list(config.get("normalized_attributes", [
#             "elevation", "w_0", "sigma", "delta", "M_x"
#         ]))
#         if "fire_map" not in _norm_attrs:
#             _norm_attrs = ["fire_map"] + _norm_attrs
#         self.normalized_attributes = _norm_attrs

#         self.initial_pos: List[int] = list(config.get("initial_pos", [15, 15]))
#         self.max_episode_steps: int = config.get("max_episode_steps", 5000)

#         movements: List[str] = list(config.get("movements", ["up", "down", "left", "right"]))
#         interactions: List[str] = list(config.get("interactions", ["fireline"]))
#         self.movements: List[str] = ["none"] + movements
#         self.interactions: List[str] = ["none"] + interactions

#         max_mitigation = max(
#             (MITIGATION_MAP.get(name, FIRELINE) for name in interactions),
#             default=FIRELINE,
#         )
#         self.sim_agent_id: int = max_mitigation + 2

#         # Count total agents (trucks) across all in-grid stations
#         self.num_agents = sum(s.get("trucks", 1) for s in self.stations)
#         if self.num_agents == 0:
#             self.num_agents = 1  # fallback single agent

#         # Actions per agent step: each agent picks (movement, interaction)
#         self.actions_per_agent = len(self.movements) * len(self.interactions)

#         # Pre-compute proportional fire spread schedule
#         self._precompute_fire_schedule(config)

#         # Build SimFire for terrain observations only
#         sim_config = FireSimInterface.build_config_dict(
#             screen_size=max(grid_rows, grid_cols),
#             terrain_type=config.get("terrain_type", "operational"),
#             latitude=config.get("latitude", self.fire_meta.get("lat_min", 39.58)),
#             longitude=config.get("longitude", self.fire_meta.get("lon_min", -121.79)),
#             resolution=config.get("resolution", 30),
#             landfire_year=config.get("landfire_year", 2020),
#             pixel_scale=config.get("pixel_scale", 12),
#             ros_attenuation=config.get("ros_attenuation", True),
#             wind_speed=config.get("wind_speed", 2),
#             wind_direction=config.get("wind_direction", 135.0),
#             moisture=config.get("moisture", 0.03),
#             fire_position_type="static",
#             max_fire_duration=100,
#         )
#         self.sim = FireSimInterface(sim_config)

#         # Normalisation bounds
#         raw_bounds = dict(self.sim.get_attribute_bounds())
#         self.min_maxes: Dict[str, Dict[str, float]] = {}
#         for attr in self.attributes:
#             if attr == "fire_map":
#                 self.min_maxes[attr] = {"min": 0, "max": float(self.sim_agent_id)}
#             elif attr in raw_bounds:
#                 self.min_maxes[attr] = {
#                     "min": float(raw_bounds[attr]["min"]),
#                     "max": float(raw_bounds[attr]["max"]),
#                 }
#             else:
#                 self.min_maxes[attr] = {"min": 0.0, "max": 1.0}

#         # Observation downsampling for memory efficiency
#         # Default 4x downsample: (101,107,6) → (26,27,6) = 4,212 floats
#         # vs full (101,107,6) = 64,842 floats. Buffer of 10k: ~337MB vs ~25GB.
#         self.obs_downsample = config.get("obs_downsample", 4)
#         # Use ceiling division to match actual slice [::obs_downsample] output size
#         import math
#         self.obs_rows = max(1, math.ceil(grid_rows / self.obs_downsample))
#         self.obs_cols = max(1, math.ceil(grid_cols / self.obs_downsample))

#         # Spaces — sequential multi-agent: one action per step for current agent
#         num_attrs = len(self.attributes)
#         self.observation_space = spaces.Box(
#             low=0.0, high=1.0,
#             shape=(self.obs_rows, self.obs_cols, num_attrs),
#             dtype=np.float32,
#         )
#         self.action_space = spaces.Discrete(self.actions_per_agent)

#         # Observation builder
#         self.obs_builder = ObservationBuilder(
#             sim_interface=self.sim,
#             attributes=self.attributes,
#             normalized_attributes=self.normalized_attributes,
#             min_maxes=self.min_maxes,
#         )

#         self.spec = EnvSpec(
#             id="DataFireEnv-v0",
#             entry_point="environment.data_fire_env:DataDrivenFireEnv",
#             max_episode_steps=self.max_episode_steps,
#         )

#         # State (set properly in reset)
#         self.agents: List[Dict[str, Any]] = []
#         self.current_agent_idx: int = 0
#         self.num_steps: int = 0
#         self.fire_timestep: int = 0
#         self.fire_sub_step: int = 0  # sub-step within current fire timestep
#         self.fire_map: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=int)
#         self.prev_burned_mask: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=bool)
#         self._is_active: bool = True

#     # ------------------------------------------------------------------
#     # Pre-computation
#     # ------------------------------------------------------------------

#     def _precompute_fire_schedule(self, config: dict) -> None:
#         """Pre-compute which cells ignite at each sub-step for proportional spread.

#         For each fire timestep t, cells with arrival==t are sorted by distance
#         from the t-1 fire perimeter and distributed evenly across sub-steps.

#         Optimized: build cumulative fire mask incrementally instead of calling
#         distance_transform_edt on the full grid each timestep.
#         """
#         self.actions_per_fire_step = config.get("actions_per_fire_step", 36)

#         self.fire_schedule: Dict[int, List[List[Tuple[int, int]]]] = {}

#         # Build sorted cell lists incrementally
#         # cumulative_fire tracks which cells have arrived up to t-1
#         cumulative_fire = np.zeros((self.fire_arrival.shape[0], self.fire_arrival.shape[1]), dtype=bool)

#         for t in range(0, self.max_fire_timestep + 1):
#             cell_coords = np.argwhere(self.fire_arrival == t)
#             if len(cell_coords) == 0:
#                 # Update cumulative (nothing new)
#                 self.fire_schedule[t] = []
#                 continue

#             if t == 0 or not np.any(cumulative_fire):
#                 # First timestep or no prior fire: all cells ignite at sub-step 0
#                 self.fire_schedule[t] = [list(map(tuple, cell_coords))]
#             else:
#                 # Sort cells by distance from existing fire front using EDT
#                 # Only compute EDT once per timestep (not per sub-step)
#                 dist = distance_transform_edt(~cumulative_fire)
#                 distances = dist[cell_coords[:, 0], cell_coords[:, 1]]
#                 order = np.argsort(distances)
#                 cell_coords = cell_coords[order]

#                 # Distribute across sub-steps
#                 n_sub = self.actions_per_fire_step
#                 batches: List[List[Tuple[int, int]]] = []
#                 if len(cell_coords) <= n_sub:
#                     for cell in cell_coords:
#                         batches.append([(int(cell[0]), int(cell[1]))])
#                 else:
#                     splits = np.array_split(cell_coords, n_sub)
#                     for split in splits:
#                         batches.append([(int(r), int(c)) for r, c in split])

#                 self.fire_schedule[t] = batches

#             # Update cumulative mask for next timestep
#             cumulative_fire[cell_coords[:, 0], cell_coords[:, 1]] = True

#     def _create_entry_stations(self, all_stations: list) -> list:
#         """Create virtual entry stations from nearest out-of-grid stations."""
#         center_lat = (self.fire_meta["lat_min"] + self.fire_meta["lat_max"]) / 2
#         center_lon = (self.fire_meta["lon_min"] + self.fire_meta["lon_max"]) / 2

#         out_of_grid = [s for s in all_stations if not s.get("in_grid", False)]
#         if not out_of_grid:
#             return [{"name": "default", "grid_row": 0, "grid_col": 0, "trucks": 1, "in_grid": True}]

#         # Sort by distance to grid center
#         for s in out_of_grid:
#             s["_dist"] = ((s["lat"] - center_lat) ** 2 + (s["lon"] - center_lon) ** 2) ** 0.5

#         out_of_grid.sort(key=lambda s: s["_dist"])
#         nearest = out_of_grid[:2]

#         entry_stations = []
#         for s in nearest:
#             # Determine direction from center
#             dlat = s["lat"] - center_lat
#             dlon = s["lon"] - center_lon

#             # Find entry row/col on grid edge
#             if abs(dlat) > abs(dlon):
#                 entry_row = 0 if dlat > 0 else self.grid_rows - 1
#                 # Map lon to col
#                 col_frac = (s["lon"] - self.fire_meta["lon_min"]) / (self.fire_meta["lon_max"] - self.fire_meta["lon_min"])
#                 entry_col = max(0, min(self.grid_cols - 1, int(col_frac * self.grid_cols)))
#             else:
#                 entry_col = 0 if dlon < 0 else self.grid_cols - 1
#                 row_frac = (self.fire_meta["lat_max"] - s["lat"]) / (self.fire_meta["lat_max"] - self.fire_meta["lat_min"])
#                 entry_row = max(0, min(self.grid_rows - 1, int(row_frac * self.grid_rows)))

#             # Snap to nearest road cell on that edge
#             if self.roads_grid is not None:
#                 entry_row, entry_col = self._snap_to_road(entry_row, entry_col)

#             entry_stations.append({
#                 "name": f"Entry from {s['name']}",
#                 "grid_row": entry_row,
#                 "grid_col": entry_col,
#                 "trucks": s.get("trucks", 1),
#                 "in_grid": True,
#             })

#         return entry_stations

#     def _snap_to_road(self, row: int, col: int, search_radius: int = 10) -> Tuple[int, int]:
#         """Find the nearest road cell to (row, col)."""
#         best_r, best_c = row, col
#         best_dist = float("inf")
#         for dr in range(-search_radius, search_radius + 1):
#             for dc in range(-search_radius, search_radius + 1):
#                 r, c = row + dr, col + dc
#                 if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
#                     if self.roads_grid[r, c] > 0:
#                         d = dr * dr + dc * dc
#                         if d < best_dist:
#                             best_dist = d
#                             best_r, best_c = r, c
#         return best_r, best_c

#     # ------------------------------------------------------------------
#     # Gymnasium interface
#     # ------------------------------------------------------------------

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[np.ndarray, Dict[str, Any]]:
#         super().reset(seed=seed)

#         # Reset SimFire (for terrain data)
#         self.sim.reset()

#         # Initialize agents at fire station positions
#         self.agents = []
#         for station in self.stations:
#             for truck_idx in range(station.get("trucks", 1)):
#                 self.agents.append({
#                     "pos": [station["grid_row"], station["grid_col"]],
#                     "station": station.get("name", "unknown"),
#                     "truck_id": truck_idx,
#                     "time_remaining": self.time_budget,
#                 })

#         # Fallback: if no stations, place single agent at initial_pos
#         if not self.agents:
#             self.agents = [{
#                 "pos": copy.copy(self.initial_pos),
#                 "station": "default",
#                 "truck_id": 0,
#                 "time_remaining": self.time_budget,
#             }]
#             self.num_agents = 1

#         self.current_agent_idx = 0
#         self.num_steps = 0
#         self.fire_timestep = 0
#         self.fire_sub_step = 0
#         self._is_active = True

#         # Build initial fire_map
#         self.fire_map = np.full((self.grid_rows, self.grid_cols), UNBURNED, dtype=int)
#         self.prev_burned_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

#         # Ignite timestep 0 cells
#         self._ignite_full_timestep(0)

#         # For compatibility, set agent_pos to first agent
#         self.agent_pos = self.agents[0]["pos"]

#         obs = self._build_observation()
#         return obs, {}

#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
#         agent = self.agents[self.current_agent_idx]

#         # Decode action
#         movement_idx = action % len(self.movements)
#         interaction_idx = action // len(self.movements)
#         movement_str = self.movements[movement_idx]
#         interaction_str = self.interactions[interaction_idx]

#         # Compute movement time cost
#         move_cost = 0.0
#         new_pos = list(agent["pos"])
#         if movement_str != "none":
#             if movement_str == "up":
#                 new_pos[0] = max(0, new_pos[0] - 1)
#             elif movement_str == "down":
#                 new_pos[0] = min(self.grid_rows - 1, new_pos[0] + 1)
#             elif movement_str == "left":
#                 new_pos[1] = max(0, new_pos[1] - 1)
#             elif movement_str == "right":
#                 new_pos[1] = min(self.grid_cols - 1, new_pos[1] + 1)

#             # Check if actually moved (not clamped at boundary)
#             if new_pos != list(agent["pos"]):
#                 on_road = self.roads_grid[new_pos[0], new_pos[1]] > 0
#                 move_cost = MOVE_TIME_ROAD if on_road else MOVE_TIME_OFFROAD

#         # Compute interaction time cost
#         build_cost = 0.0
#         if interaction_str != "none":
#             build_cost = BUILD_TIME.get(interaction_str, 45.0)

#         total_cost = move_cost + build_cost

#         # Only execute action if agent has enough time remaining
#         if total_cost <= agent["time_remaining"] + 0.01:  # small epsilon for float
#             # Move agent
#             if move_cost > 0:
#                 agent["pos"] = new_pos

#             # Place mitigation on unburned cells
#             cell_value = self.fire_map[agent["pos"][0], agent["pos"][1]]
#             if cell_value == UNBURNED and interaction_str != "none":
#                 mitigation_type = MITIGATION_MAP.get(interaction_str, FIRELINE)
#                 self.fire_map[agent["pos"][0], agent["pos"][1]] = mitigation_type

#             agent["time_remaining"] -= total_cost
#         # else: action rejected, agent stays put

#         # Advance to next agent
#         self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents

#         # When all agents have had a turn, advance fire by one sub-step
#         reward = 0.0
#         if self.current_agent_idx == 0:
#             # Save burned state before fire advance
#             self.prev_burned_mask = (self.fire_map == BURNED) | (self.fire_map == BURNING)

#             # Advance fire by one sub-step
#             self._advance_fire_substep()

#             # Compute reward on fire advance
#             total_cells = self.grid_rows * self.grid_cols
#             reward += data_driven_reward(
#                 self.fire_map, self.population_grid,
#                 self.prev_burned_mask, total_cells,
#             )

#             # Check if we should advance to next fire timestep
#             self.fire_sub_step += 1
#             batches = self.fire_schedule.get(self.fire_timestep, [])
#             if self.fire_sub_step >= max(len(batches), self.actions_per_fire_step):
#                 # Transition BURNING -> BURNED before next timestep
#                 self._transition_burning_to_burned()
#                 self.fire_timestep += 1
#                 self.fire_sub_step = 0

#                 # Reset all agents' time budgets
#                 for a in self.agents:
#                     a["time_remaining"] = self.time_budget

#                 # Check termination
#                 if self.fire_timestep > self.max_fire_timestep:
#                     self._is_active = False

#         # Timestep penalty every step
#         reward += -1000.0

#         # Update agent_pos for compatibility
#         self.agent_pos = self.agents[0]["pos"]

#         obs = self._build_observation()

#         self.num_steps += 1
#         truncated = self.num_steps >= self.max_episode_steps
#         terminated = not self._is_active

#         info = {
#             "num_steps": self.num_steps,
#             "fire_timestep": self.fire_timestep,
#             "fire_sub_step": self.fire_sub_step,
#             "current_agent": self.current_agent_idx,
#             "num_agents": self.num_agents,
#         }
#         return obs, reward, terminated, truncated, info

#     def render(self) -> None:
#         pass

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def build_terrain_base_rgb(self) -> np.ndarray:
#         """Return a (grid_rows, grid_cols, 3) uint8 RGB base frame showing:
#           - Elevation hillshade (light/shadow)
#           - Population density overlay (yellow → red warm tones)
#           - Roads overlay (white/grey lines)

#         This is computed once and can be cached; the fire/agent overlay is
#         applied on top per-frame in the GIF recorder.
#         """
#         import numpy as np

#         elev = np.load("data/camp_fire_elevation_375m.npy").astype(np.float32)
#         elev = elev[:self.grid_rows, :self.grid_cols]
#         elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)

#         # Hillshade
#         dy, dx = np.gradient(elev)
#         slope = np.sqrt(dx ** 2 + dy ** 2)
#         aspect = np.arctan2(-dx, dy)
#         altitude_rad = np.radians(45)
#         azimuth_rad = np.radians(315)
#         shade = (np.sin(altitude_rad) * np.cos(np.arctan(slope))
#                  + np.cos(altitude_rad) * np.sin(np.arctan(slope))
#                  * np.cos(azimuth_rad - aspect))
#         shade = np.clip(shade, 0.15, 1.0)  # floor at 0.15 to avoid pitch black

#         # Base colour: low=green, mid=brown, high=grey (terrain-style)
#         frame = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.float32)
#         for i in range(self.grid_rows):
#             for j in range(self.grid_cols):
#                 e = float(elev_norm[i, j])
#                 s = float(shade[i, j])
#                 # Green → olive → brown → grey gradient with elevation
#                 r = (0.35 + 0.50 * e) * s
#                 g = (0.55 - 0.25 * e) * s
#                 b = (0.25 + 0.20 * e) * s
#                 frame[i, j] = [r, g, b]

#         # Population overlay: yellow (low) → orange → red (high)
#         pop = self.population_grid[:self.grid_rows, :self.grid_cols]
#         pop_max = float(pop.max()) if pop.max() > 0 else 1.0
#         for i in range(self.grid_rows):
#             for j in range(self.grid_cols):
#                 if pop[i, j] > 0:
#                     p = float(pop[i, j]) / pop_max
#                     alpha = min(0.65, 0.25 + 0.40 * p)
#                     pr = min(1.0, 0.95 + 0.05 * p)
#                     pg = max(0.0, 0.85 - 0.65 * p)
#                     pb = 0.10
#                     frame[i, j, 0] = frame[i, j, 0] * (1 - alpha) + pr * alpha
#                     frame[i, j, 1] = frame[i, j, 1] * (1 - alpha) + pg * alpha
#                     frame[i, j, 2] = frame[i, j, 2] * (1 - alpha) + pb * alpha

#         # Roads overlay: white=primary, light-grey=secondary, grey=residential
#         road_colors_f = {
#             1: (1.0, 1.0, 1.0),
#             2: (0.78, 0.78, 0.78),
#             3: (0.65, 0.65, 0.65),
#             4: (0.55, 0.55, 0.45),
#         }
#         roads = self.roads_grid[:self.grid_rows, :self.grid_cols]
#         for i in range(self.grid_rows):
#             for j in range(self.grid_cols):
#                 rt = int(roads[i, j])
#                 if rt > 0:
#                     rc = road_colors_f.get(rt, (0.65, 0.65, 0.65))
#                     alpha = 0.70
#                     frame[i, j, 0] = frame[i, j, 0] * (1 - alpha) + rc[0] * alpha
#                     frame[i, j, 1] = frame[i, j, 1] * (1 - alpha) + rc[1] * alpha
#                     frame[i, j, 2] = frame[i, j, 2] * (1 - alpha) + rc[2] * alpha

#         return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

#     def _build_observation(self, fire_map_override=None) -> np.ndarray:
#         obs_fire_map = np.copy(self.fire_map)
#         # Stamp all agent positions
#         for i, agent in enumerate(self.agents):
#             obs_fire_map[agent["pos"][0], agent["pos"][1]] = self.sim_agent_id

#         # Get terrain data from sim and crop to grid size
#         attr_data = dict(self.sim.get_attribute_data())
#         for key in attr_data:
#             arr = np.asarray(attr_data[key], dtype=np.float32)
#             if arr.shape != (self.grid_rows, self.grid_cols):
#                 attr_data[key] = arr[:self.grid_rows, :self.grid_cols]

#         # Override fire_map
#         attr_data["fire_map"] = obs_fire_map

#         # Normalise
#         for attr in self.normalized_attributes:
#             if attr in attr_data and attr in self.min_maxes:
#                 bounds = self.min_maxes[attr]
#                 arr = attr_data[attr].astype(np.float32)
#                 denom = bounds["max"] - bounds["min"]
#                 if denom == 0:
#                     attr_data[attr] = np.zeros_like(arr)
#                 else:
#                     attr_data[attr] = np.clip((arr - bounds["min"]) / denom, 0.0, 1.0)

#         channels = []
#         for attr in self.attributes:
#             channels.append(np.asarray(attr_data[attr], dtype=np.float32))

#         full_obs = np.stack(channels, axis=-1).astype(np.float32)

#         # Downsample by taking every obs_downsample-th pixel (fastest, no blur)
#         if self.obs_downsample > 1:
#             full_obs = full_obs[::self.obs_downsample, ::self.obs_downsample, :]

#         return full_obs

#     def _ignite_full_timestep(self, t: int) -> None:
#         """Ignite all cells for a given fire timestep at once (used for t=0 only).

#         No adjacency check — these are ignition-origin cells.
#         """
#         newly_burning = (self.fire_arrival == t) & (self.fire_map == UNBURNED)
#         self.fire_map[newly_burning] = BURNING

#     def _is_fire_adjacent(self, r: int, c: int) -> bool:
#         """Return True if (r, c) is 4-connected to any BURNING or BURNED cell.

#         This makes mitigation lines (fireline/scratchline/wetline) impassable
#         barriers: fire can only spread to cells that are adjacent to active fire,
#         so a solid wall of mitigation cells blocks all propagation beyond it.
#         """
#         for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
#                 if self.fire_map[nr, nc] in (BURNING, BURNED):
#                     return True
#         return False

#     def _advance_fire_substep(self) -> None:
#         """Ignite the next batch of cells for the current fire timestep.

#         A cell ignites only if:
#           1. It is UNBURNED (not already on fire, burned, or mitigated), AND
#           2. At least one 4-connected neighbour is BURNING or BURNED.

#         This means mitigation cells (fireline/scratchline/wetline) act as
#         impassable fire barriers — fire cannot jump over or through them.
#         Cells that are only reachable through a mitigated path are protected.
#         """
#         batches = self.fire_schedule.get(self.fire_timestep, [])
#         if self.fire_sub_step < len(batches):
#             for r, c in batches[self.fire_sub_step]:
#                 if self.fire_map[r, c] == UNBURNED and self._is_fire_adjacent(r, c):
#                     self.fire_map[r, c] = BURNING

#     def _transition_burning_to_burned(self) -> None:
#         """Transition all BURNING cells to BURNED."""
#         was_burning = self.fire_map == BURNING
#         self.fire_map[was_burning] = BURNED


"""Data-driven firefighting RL environment using real Camp Fire VIIRS data.

Fire spread follows the precomputed fire_arrival grid derived from VIIRS
satellite observations (FEDS dataset). Fire advances proportionally — cells
are distributed evenly across sub-steps within each fire timestep to avoid
visual/temporal jumps.

Multi-agent support: multiple fire engines start from real fire station
locations. Each agent has a time budget per fire timestep (180 min for
3-hour steps). Movement on roads is fast (5 min/cell), off-road is slow
(15 min/cell), and building mitigation consumes significant time.

This version adds:
1. Extra observation channels so the state is closer to Markov:
   - current agent position
   - current agent time remaining
   - fire timestep
   - fire sub-step
2. Dense reward shaping:
   - reward for useful mitigation placement
   - extra reward if mitigation is adjacent to fire
   - small shaping reward for moving toward the nearest burning cell
   - penalty for rejected actions
   - penalty for useless no-op actions
"""
import copy
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from scipy.ndimage import distance_transform_edt

from environment.base_env import BaseFireEnv
from fire_sim.sim_interface import FireSimInterface
from reward.reward_functions import data_driven_reward

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
MOVE_TIME_ROAD = 5.0
MOVE_TIME_OFFROAD = 15.0
BUILD_TIME: Dict[str, float] = {
    "fireline": 45.0,
    "scratchline": 30.0,
    "wetline": 20.0,
}


class DataDrivenFireEnv(BaseFireEnv):
    """Gymnasium environment using real fire spread data from VIIRS observations."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.screen_size: int = config.get("screen_size", 250)

        # ------------------------------------------------------------------
        # Load fire arrival data
        # ------------------------------------------------------------------
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
        self.time_budget = self.hours_per_timestep * 60.0

        # ------------------------------------------------------------------
        # Roads / population / stations
        # ------------------------------------------------------------------
        roads_path = data_dir / config.get("roads_file", "camp_fire_roads_375m.npy")
        if roads_path.exists():
            self.roads_grid = np.load(roads_path)
        else:
            self.roads_grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)

        pop_path = data_dir / config.get(
            "population_file", "camp_fire_population_375m.npy"
        )
        if pop_path.exists():
            self.population_grid = np.load(pop_path).astype(np.float32)
        else:
            self.population_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        stations_path = data_dir / config.get(
            "stations_file", "camp_fire_stations.json"
        )
        if stations_path.exists():
            with open(stations_path) as f:
                all_stations = json.load(f)
            self.stations = [s for s in all_stations if s.get("in_grid", False)]
        else:
            self.stations = []

        if not self.stations and stations_path.exists():
            with open(stations_path) as f:
                all_stations = json.load(f)
            self.stations = self._create_entry_stations(all_stations)

        # ------------------------------------------------------------------
        # Config
        # ------------------------------------------------------------------
        self.attributes: List[str] = list(
            config.get(
                "attributes",
                ["fire_map", "elevation", "w_0", "sigma", "delta", "M_x"],
            )
        )

        _norm_attrs: List[str] = list(
            config.get(
                "normalized_attributes",
                ["elevation", "w_0", "sigma", "delta", "M_x"],
            )
        )
        if "fire_map" not in _norm_attrs:
            _norm_attrs = ["fire_map"] + _norm_attrs
        self.normalized_attributes = _norm_attrs

        self.initial_pos: List[int] = list(config.get("initial_pos", [15, 15]))
        self.max_episode_steps: int = config.get("max_episode_steps", 5000)

        movements: List[str] = list(
            config.get("movements", ["up", "down", "left", "right"])
        )
        interactions: List[str] = list(
            config.get("interactions", ["fireline"])
        )

        self.movements: List[str] = ["none"] + movements
        self.interactions: List[str] = ["none"] + interactions

        max_mitigation = max(
            (MITIGATION_MAP.get(name, FIRELINE) for name in interactions),
            default=FIRELINE,
        )
        self.sim_agent_id: int = max_mitigation + 2

        self.num_agents = sum(s.get("trucks", 1) for s in self.stations)
        if self.num_agents == 0:
            self.num_agents = 1
        max_agents = config.get("max_agents", None)
        if max_agents is not None:
            self.num_agents = min(self.num_agents, int(max_agents))

        self.actions_per_agent = len(self.movements) * len(self.interactions)

        # ------------------------------------------------------------------
        # Reward shaping config
        # ------------------------------------------------------------------
        self.step_penalty = float(config.get("step_penalty", -100.0))
        self.invalid_action_penalty = float(config.get("invalid_action_penalty", -20.0))
        self.useless_noop_penalty = float(config.get("useless_noop_penalty", -2.0))
        self.mitigation_reward = float(config.get("mitigation_reward", 5.0))
        # Fallback reward for placing mitigation right at the burning edge.
        # Kept small so the dominant signal is ahead_mitigation_reward.
        self.adjacent_mitigation_reward = float(
            config.get("adjacent_mitigation_reward", 10.0)
        )
        # Primary mitigation reward: cell is on the projected spread path
        # (fire_arrival within +4 fire timesteps of current).
        self.ahead_mitigation_reward = float(
            config.get("ahead_mitigation_reward", 40.0)
        )
        # Extra reward per adjacent mitigation neighbor — drives line-building.
        self.line_connectivity_reward = float(
            config.get("line_connectivity_reward", 25.0)
        )
        self.move_toward_fire_reward = float(
            config.get("move_toward_fire_reward", 1.0)
        )
        self.move_away_fire_penalty = float(
            config.get("move_away_fire_penalty", -1.0)
        )
        self.revisit_penalty = float(config.get("revisit_penalty", 0.0))
        self.revisit_window = int(config.get("revisit_window", 20))

        # ------------------------------------------------------------------
        # Fire schedule
        # ------------------------------------------------------------------
        self._precompute_fire_schedule(config)

        # ------------------------------------------------------------------
        # SimFire for terrain observations only
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Normalization bounds
        # ------------------------------------------------------------------
        raw_bounds = dict(self.sim.get_attribute_bounds())
        self.min_maxes: Dict[str, Dict[str, float]] = {}
        for attr in self.attributes:
            if attr == "fire_map":
                self.min_maxes[attr] = {"min": 0, "max": float(self.sim_agent_id)}
            elif attr in raw_bounds:
                self.min_maxes[attr] = {
                    "min": float(raw_bounds[attr]["min"]),
                    "max": float(raw_bounds[attr]["max"]),
                }
            else:
                self.min_maxes[attr] = {"min": 0.0, "max": 1.0}

        # ------------------------------------------------------------------
        # Downsampling
        # ------------------------------------------------------------------
        self.obs_downsample = config.get("obs_downsample", 4)
        self.obs_rows = max(1, math.ceil(grid_rows / self.obs_downsample))
        self.obs_cols = max(1, math.ceil(grid_cols / self.obs_downsample))

        # Extra state channels:
        # 1. current agent position map
        # 2. current agent time remaining map
        # 3. fire timestep map
        # 4. fire sub-step map
        # 5. fire arrival urgency map
        self.extra_state_channels = 5

        num_attrs = len(self.attributes) + self.extra_state_channels
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_rows, self.obs_cols, num_attrs),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.actions_per_agent)

        self.spec = EnvSpec(
            id="DataFireEnv-v0",
            entry_point="environment.data_fire_env:DataDrivenFireEnv",
            max_episode_steps=self.max_episode_steps,
        )

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self.agents: List[Dict[str, Any]] = []
        self.current_agent_idx: int = 0
        self.num_steps: int = 0
        self.fire_timestep: int = 0
        self.fire_sub_step: int = 0
        self.fire_map: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=int)
        self.prev_burned_mask: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=bool)
        self._is_active: bool = True

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_fire_schedule(self, config: dict) -> None:
        """Pre-compute which cells ignite at each sub-step for proportional spread."""
        self.actions_per_fire_step = config.get("actions_per_fire_step", 36)
        self.fire_schedule: Dict[int, List[List[Tuple[int, int]]]] = {}

        cumulative_fire = np.zeros(
            (self.fire_arrival.shape[0], self.fire_arrival.shape[1]),
            dtype=bool,
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
                sorted_coords = cell_coords[order]

                n_sub = self.actions_per_fire_step
                batches: List[List[Tuple[int, int]]] = []
                if len(sorted_coords) <= n_sub:
                    for cell in sorted_coords:
                        batches.append([(int(cell[0]), int(cell[1]))])
                else:
                    splits = np.array_split(sorted_coords, n_sub)
                    for split in splits:
                        batches.append([(int(r), int(c)) for r, c in split])

                self.fire_schedule[t] = batches
                cell_coords = sorted_coords

            cumulative_fire[cell_coords[:, 0], cell_coords[:, 1]] = True

    def _create_entry_stations(self, all_stations: list) -> list:
        """Create virtual entry stations from nearest out-of-grid stations."""
        center_lat = (self.fire_meta["lat_min"] + self.fire_meta["lat_max"]) / 2
        center_lon = (self.fire_meta["lon_min"] + self.fire_meta["lon_max"]) / 2

        out_of_grid = [s for s in all_stations if not s.get("in_grid", False)]
        if not out_of_grid:
            return [{
                "name": "default",
                "grid_row": 0,
                "grid_col": 0,
                "trucks": 1,
                "in_grid": True,
            }]

        for s in out_of_grid:
            s["_dist"] = ((s["lat"] - center_lat) ** 2 + (s["lon"] - center_lon) ** 2) ** 0.5

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
                    0,
                    min(self.grid_cols - 1, int(col_frac * self.grid_cols)),
                )
            else:
                entry_col = 0 if dlon < 0 else self.grid_cols - 1
                row_frac = (self.fire_meta["lat_max"] - s["lat"]) / (
                    self.fire_meta["lat_max"] - self.fire_meta["lat_min"]
                )
                entry_row = max(
                    0,
                    min(self.grid_rows - 1, int(row_frac * self.grid_rows)),
                )

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

        self.sim.reset()

        self.agents = []
        for station in self.stations:
            for truck_idx in range(station.get("trucks", 1)):
                if len(self.agents) >= self.num_agents:
                    break
                self.agents.append({
                    "pos": [station["grid_row"], station["grid_col"]],
                    "station": station.get("name", "unknown"),
                    "truck_id": truck_idx,
                    "time_remaining": self.time_budget,
                    "recent_positions": deque(maxlen=self.revisit_window),
                })
            if len(self.agents) >= self.num_agents:
                break

        if not self.agents:
            self.agents = [{
                "pos": copy.copy(self.initial_pos),
                "station": "default",
                "truck_id": 0,
                "time_remaining": self.time_budget,
                "recent_positions": deque(maxlen=self.revisit_window),
            }]
            self.num_agents = 1

        self.current_agent_idx = 0
        self.num_steps = 0
        self.fire_timestep = 0
        self.fire_sub_step = 0
        self._is_active = True

        self.fire_map = np.full((self.grid_rows, self.grid_cols), UNBURNED, dtype=int)
        self.prev_burned_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

        self._ignite_full_timestep(0)

        self.agent_pos = self.agents[0]["pos"]
        obs = self._build_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        agent = self.agents[self.current_agent_idx]
        old_pos = list(agent["pos"])
        old_dist_to_fire = self._distance_to_next_front(old_pos)

        movement_idx = action % len(self.movements)
        interaction_idx = action // len(self.movements)
        movement_str = self.movements[movement_idx]
        interaction_str = self.interactions[interaction_idx]

        reward = 0.0

        move_cost = 0.0
        new_pos = list(agent["pos"])
        wall_hit = False

        if movement_str != "none":
            if movement_str == "up":
                new_pos[0] = max(0, new_pos[0] - 1)
            elif movement_str == "down":
                new_pos[0] = min(self.grid_rows - 1, new_pos[0] + 1)
            elif movement_str == "left":
                new_pos[1] = max(0, new_pos[1] - 1)
            elif movement_str == "right":
                new_pos[1] = min(self.grid_cols - 1, new_pos[1] + 1)

            if new_pos != list(agent["pos"]):
                on_road = self.roads_grid[new_pos[0], new_pos[1]] > 0
                move_cost = MOVE_TIME_ROAD if on_road else MOVE_TIME_OFFROAD
            else:
                wall_hit = True

        build_cost = 0.0
        if interaction_str != "none":
            build_cost = BUILD_TIME.get(interaction_str, 45.0)

        total_cost = move_cost + build_cost
        action_executed = False
        placed_mitigation = False

        if wall_hit:
            reward += self.invalid_action_penalty
        elif total_cost <= agent["time_remaining"] + 1e-6:
            action_executed = True

            if move_cost > 0:
                agent["pos"] = new_pos

            cell_value = self.fire_map[agent["pos"][0], agent["pos"][1]]
            if cell_value == UNBURNED and interaction_str != "none":
                mitigation_type = MITIGATION_MAP.get(interaction_str, FIRELINE)
                r, c = agent["pos"][0], agent["pos"][1]
                arrival_at_cell = int(self.fire_arrival[r, c])
                is_ahead_of_fire = (
                    self.fire_timestep < arrival_at_cell <= self.fire_timestep + 4
                )
                adjacent_to_fire = self._is_burning_adjacent(r, c)
                self.fire_map[r, c] = mitigation_type
                placed_mitigation = True

                reward += self.mitigation_reward
                if is_ahead_of_fire:
                    reward += self.ahead_mitigation_reward
                elif adjacent_to_fire:
                    # Fallback: still useful, but smaller incentive than ahead placement
                    reward += self.adjacent_mitigation_reward

                # Connectivity bonus: reward each adjacent cell that is already mitigation
                for _dr, _dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + _dr, c + _dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        if self.fire_map[nr, nc] in (FIRELINE, SCRATCHLINE, WETLINE):
                            reward += self.line_connectivity_reward

            agent["time_remaining"] -= total_cost
        else:
            # Budget exhausted: silently skip the action.  The fire spreading
            # already encodes the opportunity cost of wasted time; piling on
            # invalid_action_penalty here floods the episode with -2 hits for
            # every one of the ~30 budget-exhausted steps per fire window,
            # drowning out all mitigation reward signal.
            pass

        # Movement shaping (distance to projected spread front, not current flames)
        new_dist_to_fire = self._distance_to_next_front(agent["pos"])
        if move_cost > 0 and action_executed:
            if new_dist_to_fire < old_dist_to_fire:
                reward += self.move_toward_fire_reward
            elif new_dist_to_fire > old_dist_to_fire:
                reward += self.move_away_fire_penalty

            # Revisit penalty: discourage A→B→A oscillation loops.
            new_pos_tuple = tuple(agent["pos"])
            if new_pos_tuple in agent["recent_positions"]:
                reward += self.revisit_penalty
            agent["recent_positions"].append(new_pos_tuple)

        # Penalize useless no-op only when the agent still has time to spend.
        # When budget is already 0, no-op is the correct behavior — penalizing
        # it forces the agent to prefer rejected move attempts (-2) over doing
        # nothing, which is exactly the "hover/oscillate" loop we want to break.
        if movement_str == "none" and interaction_str == "none":
            if old_dist_to_fire > 0 and agent["time_remaining"] > 0:
                reward += self.useless_noop_penalty

        # Penalize asking to build but failing to place (cell already burning/burned)
        if interaction_str != "none" and action_executed and not placed_mitigation:
            reward += self.invalid_action_penalty

        # Advance to next agent
        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents

        # After all agents have acted, advance fire once
        if self.current_agent_idx == 0:
            self.prev_burned_mask = (self.fire_map == BURNED) | (self.fire_map == BURNING)

            self._advance_fire_substep()

            total_cells = self.grid_rows * self.grid_cols
            reward += data_driven_reward(
                self.fire_map,
                self.population_grid,
                self.prev_burned_mask,
                total_cells,
            )

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

        reward += self.step_penalty

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
        """Return a terrain RGB frame for GIF rendering."""
        elev = np.load("data/camp_fire_elevation_375m.npy").astype(np.float32)
        elev = elev[:self.grid_rows, :self.grid_cols]
        elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)

        dy, dx = np.gradient(elev)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        aspect = np.arctan2(-dx, dy)
        altitude_rad = np.radians(45)
        azimuth_rad = np.radians(315)
        shade = (
            np.sin(altitude_rad) * np.cos(np.arctan(slope))
            + np.cos(altitude_rad) * np.sin(np.arctan(slope))
            * np.cos(azimuth_rad - aspect)
        )
        shade = np.clip(shade, 0.15, 1.0)

        frame = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.float32)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                e = float(elev_norm[i, j])
                s = float(shade[i, j])
                r = (0.35 + 0.50 * e) * s
                g = (0.55 - 0.25 * e) * s
                b = (0.25 + 0.20 * e) * s
                frame[i, j] = [r, g, b]

        pop = self.population_grid[:self.grid_rows, :self.grid_cols]
        pop_max = float(pop.max()) if pop.max() > 0 else 1.0
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if pop[i, j] > 0:
                    p = float(pop[i, j]) / pop_max
                    alpha = min(0.65, 0.25 + 0.40 * p)
                    pr = min(1.0, 0.95 + 0.05 * p)
                    pg = max(0.0, 0.85 - 0.65 * p)
                    pb = 0.10
                    frame[i, j, 0] = frame[i, j, 0] * (1 - alpha) + pr * alpha
                    frame[i, j, 1] = frame[i, j, 1] * (1 - alpha) + pg * alpha
                    frame[i, j, 2] = frame[i, j, 2] * (1 - alpha) + pb * alpha

        road_colors_f = {
            1: (1.0, 1.0, 1.0),
            2: (0.78, 0.78, 0.78),
            3: (0.65, 0.65, 0.65),
            4: (0.55, 0.55, 0.45),
        }
        roads = self.roads_grid[:self.grid_rows, :self.grid_cols]
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                rt = int(roads[i, j])
                if rt > 0:
                    rc = road_colors_f.get(rt, (0.65, 0.65, 0.65))
                    alpha = 0.70
                    frame[i, j, 0] = frame[i, j, 0] * (1 - alpha) + rc[0] * alpha
                    frame[i, j, 1] = frame[i, j, 1] * (1 - alpha) + rc[1] * alpha
                    frame[i, j, 2] = frame[i, j, 2] * (1 - alpha) + rc[2] * alpha

        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    def _build_observation(self, fire_map_override=None) -> np.ndarray:
        """Build observation with base channels + extra state channels."""
        obs_fire_map = np.copy(self.fire_map)

        for agent in self.agents:
            obs_fire_map[agent["pos"][0], agent["pos"][1]] = self.sim_agent_id

        attr_data = dict(self.sim.get_attribute_data())
        for key in attr_data:
            arr = np.asarray(attr_data[key], dtype=np.float32)
            if arr.shape != (self.grid_rows, self.grid_cols):
                attr_data[key] = arr[:self.grid_rows, :self.grid_cols]

        attr_data["fire_map"] = obs_fire_map

        for attr in self.normalized_attributes:
            if attr in attr_data and attr in self.min_maxes:
                bounds = self.min_maxes[attr]
                arr = attr_data[attr].astype(np.float32)
                denom = bounds["max"] - bounds["min"]
                if denom == 0:
                    attr_data[attr] = np.zeros_like(arr)
                else:
                    attr_data[attr] = np.clip(
                        (arr - bounds["min"]) / denom,
                        0.0,
                        1.0,
                    )

        channels = []
        for attr in self.attributes:
            channels.append(np.asarray(attr_data[attr], dtype=np.float32))

        # --------------------------------------------------------------
        # Extra channels
        # --------------------------------------------------------------
        current_agent = self.agents[self.current_agent_idx]
        current_agent_pos_map = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        current_agent_pos_map[current_agent["pos"][0], current_agent["pos"][1]] = 1.0

        time_remaining_val = float(current_agent["time_remaining"]) / max(self.time_budget, 1.0)
        time_remaining_map = np.full(
            (self.grid_rows, self.grid_cols),
            np.clip(time_remaining_val, 0.0, 1.0),
            dtype=np.float32,
        )

        fire_timestep_val = float(self.fire_timestep) / max(self.max_fire_timestep, 1)
        fire_timestep_map = np.full(
            (self.grid_rows, self.grid_cols),
            np.clip(fire_timestep_val, 0.0, 1.0),
            dtype=np.float32,
        )

        fire_sub_step_val = float(self.fire_sub_step) / max(self.actions_per_fire_step, 1)
        fire_sub_step_map = np.full(
            (self.grid_rows, self.grid_cols),
            np.clip(fire_sub_step_val, 0.0, 1.0),
            dtype=np.float32,
        )

        # Fire arrival urgency: 1.0 = burns next timestep, decays linearly to 0
        # over an 8-timestep lookahead window.  Already-burned cells → 0.
        steps_until_burn = (
            self.fire_arrival[:self.grid_rows, :self.grid_cols].astype(np.float32)
            - float(self.fire_timestep)
        )
        _lookahead = 8.0
        arrival_channel = np.where(
            steps_until_burn > 0,
            np.clip(1.0 - steps_until_burn / _lookahead, 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        channels.append(current_agent_pos_map)
        channels.append(time_remaining_map)
        channels.append(fire_timestep_map)
        channels.append(fire_sub_step_map)
        channels.append(arrival_channel)

        full_obs = np.stack(channels, axis=-1).astype(np.float32)

        if self.obs_downsample > 1:
            full_obs = full_obs[::self.obs_downsample, ::self.obs_downsample, :]

        return full_obs

    def _ignite_full_timestep(self, t: int) -> None:
        """Ignite all cells for a given fire timestep at once (used for t=0 only)."""
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

    def _is_burning_adjacent(self, r: int, c: int) -> bool:
        """Return True if (r, c) is 4-connected to any BURNING cell."""
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                if self.fire_map[nr, nc] == BURNING:
                    return True
        return False

    def _nearest_burning_distance(self, pos: list) -> float:
        """Manhattan distance from pos to nearest burning cell."""
        burning = np.argwhere(self.fire_map == BURNING)
        if len(burning) == 0:
            return 0.0
        dists = np.abs(burning[:, 0] - pos[0]) + np.abs(burning[:, 1] - pos[1])
        return float(np.min(dists))

    def _distance_to_next_front(self, pos: list) -> float:
        """Manhattan distance to the projected spread front.

        Targets unburned cells scheduled to ignite within the next 1–3 fire
        timesteps so movement shaping points the agent ahead of the fire, not
        at cells already burning.  Falls back to nearest burning cell when no
        such cells exist (end of fire or all consumed).
        """
        lookahead_mask = (
            (self.fire_arrival >= self.fire_timestep + 1)
            & (self.fire_arrival <= self.fire_timestep + 3)
            & (self.fire_map == UNBURNED)
        )
        front_cells = np.argwhere(lookahead_mask)
        if len(front_cells) == 0:
            return self._nearest_burning_distance(pos)
        dists = np.abs(front_cells[:, 0] - pos[0]) + np.abs(front_cells[:, 1] - pos[1])
        return float(np.min(dists))

    def _advance_fire_substep(self) -> None:
        """Ignite the next batch of cells for the current fire timestep."""
        batches = self.fire_schedule.get(self.fire_timestep, [])
        if self.fire_sub_step < len(batches):
            for r, c in batches[self.fire_sub_step]:
                if self.fire_map[r, c] == UNBURNED and self._is_fire_adjacent(r, c):
                    self.fire_map[r, c] = BURNING

    def _transition_burning_to_burned(self) -> None:
        """Transition all BURNING cells to BURNED."""
        was_burning = self.fire_map == BURNING
        self.fire_map[was_burning] = BURNED
