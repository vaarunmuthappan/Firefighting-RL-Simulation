"""Terrain loading utilities for the fire simulation.

Supports functional terrain (Perlin noise + chaparral fuel) and operational
terrain (LANDFIRE data downloaded for a given lat/lon).
"""
from typing import Any


class TerrainLoader:
    """Loads terrain configuration for SimFire.

    Provides factory methods for functional and operational terrain so that
    the caller does not need to know the nested SimFire config structure.
    """

    @staticmethod
    def load_functional() -> dict:
        """Return the functional terrain config sub-dict.

        Uses Perlin-noise elevation and chaparral fuel model -- no external
        data downloads required.

        Returns:
            A dict for the "terrain" key of the SimFire config.
        """
        return {
            "topography": {
                "type": "functional",
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
                },
            },
            "fuel": {
                "type": "functional",
                "functional": {
                    "function": "chaparral",
                    "chaparral": {"seed": 1113},
                },
            },
        }

    @staticmethod
    def load_operational(
        lat: float, lon: float, year: int, data_path: str = ""
    ) -> dict:
        """Return operational terrain config using LANDFIRE data.

        SimFire handles the actual LANDFIRE download and processing when
        terrain type is set to "operational". This method returns the terrain
        config dict with the correct type setting.

        Args:
            lat: Latitude of the area of interest.
            lon: Longitude of the area of interest.
            year: LANDFIRE data year.
            data_path: Local path to pre-downloaded LANDFIRE files (unused;
                SimFire downloads data automatically).

        Returns:
            A terrain config dict with type set to "operational".
        """
        return {
            "topography": {"type": "operational"},
            "fuel": {"type": "operational"},
        }

    @staticmethod
    def get_config_dict(terrain_type: str, **kwargs: Any) -> dict:
        """Dispatch to the appropriate terrain loader based on terrain_type.

        Args:
            terrain_type: "functional" or "operational".
            **kwargs: Additional keyword arguments forwarded to the loader.

        Returns:
            A terrain config dict suitable for the SimFire config.

        Raises:
            ValueError: If an unrecognised terrain_type is provided.
        """
        if terrain_type == "functional":
            return TerrainLoader.load_functional()
        elif terrain_type == "operational":
            return TerrainLoader.load_operational(**kwargs)
        else:
            raise ValueError(
                f"Unknown terrain_type '{terrain_type}'. "
                "Expected 'functional' or 'operational'."
            )
