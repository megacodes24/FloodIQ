from __future__ import annotations

import numpy as np

from floodiq.models import GridData, RainScenario


def scenario_feature_stack(grid: GridData, scenario: RainScenario) -> np.ndarray:
    rows, cols = grid.elevation.shape
    rainfall = np.full((rows, cols), scenario.rainfall_inches_per_hour, dtype=np.float32)
    duration = np.full((rows, cols), scenario.duration_hours, dtype=np.float32)
    timestep = np.full((rows, cols), scenario.timestep_seconds, dtype=np.float32)

    drain_scale = np.max(grid.drain_capacity) or 1.0
    infiltration_scale = np.max(grid.infiltration_rate) or 1.0

    stacked = np.stack(
        [
            grid.elevation.astype(np.float32),
            (grid.infiltration_rate / infiltration_scale).astype(np.float32),
            (grid.drain_capacity / drain_scale).astype(np.float32),
            grid.building_mask.astype(np.float32),
            grid.vulnerability.astype(np.float32),
            rainfall,
            duration,
            timestep,
        ],
        axis=0,
    )
    return stacked
