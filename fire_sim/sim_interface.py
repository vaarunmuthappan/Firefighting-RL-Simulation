"""Thin wrapper around FireSimulation providing a standardized interface.

This module isolates the SimFire dependency so that the simulation backend
can be swapped by only modifying this file.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config


class FireSimInterface:
    """Standardized interface wrapping FireSimulation.

    Provides stable method signatures independent of SimFire internals so
    that the rest of the codebase does not directly depend on SimFire's API.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the FireSimulation from a config dictionary.

        Args:
            config: Full SimFire config dict as expected by Config(config_dict=...).
        """
        self._config = config
        self._sim = FireSimulation(Config(config_dict=config))

    def reset(self) -> dict:
        """Reset the simulation to initial conditions.

        Returns:
            Attribute data dict from sim.get_attribute_data() after reset.
        """
        self._sim.reset()
        return self._sim.get_attribute_data()

    def step(self) -> Tuple[np.ndarray, bool]:
        """Advance the simulation by one timestep.

        Returns:
            A tuple of (fire_map, is_active) where fire_map is a 2D numpy
            array and is_active is False when the fire is done spreading.
        """
        fire_map, is_active = self._sim.run(1)
        return fire_map, is_active

    def apply_mitigation(self, x: int, y: int, mitigation_type: int = 3) -> None:
        """Place a mitigation action at the given map coordinates.

        Args:
            x: Column coordinate (note: SimFire uses (col, row, agent_id) ordering).
            y: Row coordinate.
            mitigation_type: Integer BurnStatus value (3=FIRELINE, 4=SCRATCHLINE, 5=WETLINE).
        """
        self._sim.update_mitigation([(x, y, mitigation_type)])

    def update_agent_position(self, x: int, y: int, agent_id: int = 0) -> None:
        """Update the agent's displayed position in the simulation.

        Args:
            x: Column coordinate.
            y: Row coordinate.
            agent_id: Agent identifier (0 for single-agent).
        """
        self._sim.update_agent_positions([(x, y, agent_id)])

    def get_attribute_data(self) -> dict:
        """Return raw attribute arrays from the simulation.

        Returns:
            OrderedDict mapping attribute names to numpy arrays.
        """
        return self._sim.get_attribute_data()

    def get_attribute_bounds(self) -> dict:
        """Return min/max bounds for each simulation attribute.

        Returns:
            OrderedDict mapping attribute names to {"min": v, "max": v} dicts.
        """
        return self._sim.get_attribute_bounds()

    def get_available_actions(self) -> dict:
        """Return the available mitigation actions from the simulation.

        Returns:
            Dict mapping action name strings to IntEnum values.
        """
        return self._sim.get_actions()

    def save_gif(self, path: str) -> None:
        """Save a GIF of the current episode to disk.

        Args:
            path: Directory path where the GIF will be saved.
        """
        self._sim.save_gif(path)

    @property
    def fire_map(self) -> np.ndarray:
        """Current 2D fire map as a numpy array."""
        return self._sim.fire_map

    @property
    def rate_of_spread(self) -> np.ndarray:
        """Current 2D rate-of-spread grid (ft/min per cell) from the fire manager."""
        return self._sim.fire_manager.rate_of_spread

    @property
    def rendering(self) -> bool:
        """Whether pygame rendering is enabled."""
        return self._sim.rendering

    @rendering.setter
    def rendering(self, value: bool) -> None:
        self._sim.rendering = value

    @property
    def screen_size(self) -> int:
        """Map width (= height for square maps) in pixels."""
        ss = self._sim.config.area.screen_size
        # Config stores screen_size as a (width, height) tuple; return width.
        if isinstance(ss, (tuple, list)):
            return int(ss[0])
        return int(ss)

    @property
    def config(self):
        """The underlying SimFire Config object."""
        return self._sim.config

    @staticmethod
    def build_config_dict(
        screen_size: int = 64,
        wind_speed: float = 2,
        wind_direction: float = 135.0,
        moisture: float = 0.03,
        fire_position_type: str = "random",
        terrain_type: str = "functional",
        max_fire_duration: int = 4,
        pixel_scale: int = 12,
        ros_attenuation: bool = False,
        latitude: float = 33.03,
        longitude: float = -116.66,
        resolution: int = 30,
        landfire_year: int = 2020,
    ) -> dict:
        """Build the full SimFire config dictionary.

        Args:
            screen_size: Side length of the square simulation grid.
            wind_speed: Wind speed in m/s.
            wind_direction: Wind direction in degrees.
            moisture: Dead fuel moisture content (0-1).
            fire_position_type: "random" or "static".
            terrain_type: "functional" (no download) or "operational" (LANDFIRE).
            max_fire_duration: Maximum number of simulation timesteps.
            pixel_scale: Real-world meters per pixel.
            ros_attenuation: Enable per-mitigation-type rate-of-spread reduction.
            latitude: Latitude for operational LANDFIRE data.
            longitude: Longitude for operational LANDFIRE data.
            resolution: LANDFIRE resolution in meters (only 30 supported).
            landfire_year: LANDFIRE data year (2019, 2020, or 2022).

        Returns:
            A dict suitable for Config(config_dict=...).
        """
        topo_type = terrain_type
        fuel_type = terrain_type

        return {
            "area": {
                "screen_size": [screen_size, screen_size],
                "pixel_scale": pixel_scale,
            },
            "display": {
                "fire_size": 2,
                "control_line_size": 2,
                "agent_size": 4,
            },
            "simulation": {
                "update_rate": 1,
                "runtime": "24h",
                "headless": True,
                "record": False,
                "save_data": False,
                "draw_spread_graph": False,
                "data_type": "npy",
                "sf_home": "/tmp/simfire_outputs",
            },
            "mitigation": {"ros_attenuation": ros_attenuation},
            "operational": {
                "seed": None,
                "latitude": latitude,
                "longitude": longitude,
                "height": screen_size * resolution,  # SimFire expects meters
                "width": screen_size * resolution,   # SimFire expects meters
                "resolution": resolution,
                "year": landfire_year,
            },
            "terrain": {
                "topography": {
                    "type": topo_type,
                    "functional": {
                        "function": "perlin",
                        "perlin": {
                            "octaves": 3,
                            "persistence": 0.7,
                            "lacunarity": 2.0,
                            "seed": 827,
                            "range_min": 100.0,
                            "range_max": 300.0,
                        },
                        "gaussian": {
                            "amplitude": 500,
                            "mu_x": 50,
                            "mu_y": 50,
                            "sigma_x": 50,
                            "sigma_y": 50,
                        },
                    },
                },
                "fuel": {
                    "type": fuel_type,
                    "functional": {
                        "function": "chaparral",
                        "chaparral": {"seed": 1113},
                    },
                },
            },
            "fire": {
                "fire_initial_position": {
                    "type": fire_position_type,
                    "static": {"position": (screen_size // 2, screen_size // 2)},
                    "random": {"seed": 1234},
                },
                "max_fire_duration": max_fire_duration,
                "diagonal_spread": True,
            },
            "environment": {"moisture": moisture},
            "wind": {
                "function": "simple",
                "cfd": {
                    "time_to_train": 1000,
                    "iterations": 1,
                    "scale": 1,
                    "timestep_dt": 1.0,
                    "diffusion": 0.0,
                    "viscosity": 0.0000001,
                    "speed": 19,
                    "direction": "north",
                },
                "simple": {"speed": wind_speed, "direction": wind_direction},
                "perlin": {
                    "speed": {
                        "seed": 2345,
                        "scale": 400,
                        "octaves": 3,
                        "persistence": 0.7,
                        "lacunarity": 2.0,
                        "range_min": 7,
                        "range_max": 47,
                    },
                    "direction": {
                        "seed": 650,
                        "scale": 1500,
                        "octaves": 2,
                        "persistence": 0.9,
                        "lacunarity": 1.0,
                        "range_min": 0.0,
                        "range_max": 360.0,
                    },
                },
            },
        }
