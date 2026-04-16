from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RainScenario:
    name: str
    rainfall_inches_per_hour: float
    duration_hours: float
    timestep_seconds: float = 20.0

    @property
    def rainfall_meters_per_second(self) -> float:
        inches_to_meters = 0.0254
        return (self.rainfall_inches_per_hour * inches_to_meters) / 3600.0

    @property
    def total_steps(self) -> int:
        return max(1, int((self.duration_hours * 3600.0) / self.timestep_seconds))


@dataclass(frozen=True)
class GridData:
    elevation: np.ndarray
    infiltration_rate: np.ndarray
    drain_capacity: np.ndarray
    building_mask: np.ndarray
    vulnerability: np.ndarray
    block_ids: np.ndarray
    block_names: list[str]


@dataclass(frozen=True)
class SimulationOutput:
    water_depth_m: np.ndarray
    max_water_depth_m: np.ndarray
    risk_score: np.ndarray
    water_volume_m3: float
    flooded_cells: int
