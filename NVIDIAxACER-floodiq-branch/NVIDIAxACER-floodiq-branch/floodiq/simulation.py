from __future__ import annotations

import numpy as np

from .models import GridData, RainScenario, SimulationOutput


class FloodSimulator:
    """A lightweight shallow-water-inspired grid solver for hackathon MVPs."""

    def __init__(self, cell_size_m: float = 20.0, lateral_flow_rate: float = 0.18) -> None:
        self.cell_size_m = cell_size_m
        self.lateral_flow_rate = lateral_flow_rate

    def run(self, grid: GridData, scenario: RainScenario) -> SimulationOutput:
        water = np.zeros_like(grid.elevation, dtype=float)
        max_depth = np.zeros_like(grid.elevation, dtype=float)
        active_cells = ~grid.building_mask
        cell_area = self.cell_size_m ** 2

        for _ in range(scenario.total_steps):
            rainfall_gain = scenario.rainfall_meters_per_second * scenario.timestep_seconds
            water[active_cells] += rainfall_gain

            infiltration_loss = np.minimum(
                water,
                grid.infiltration_rate * scenario.timestep_seconds,
            )
            water -= infiltration_loss

            drain_loss = np.minimum(water, grid.drain_capacity * scenario.timestep_seconds)
            water -= drain_loss

            water = self._redistribute(water, grid.elevation, grid.building_mask)
            water[~active_cells] = 0.0
            max_depth = np.maximum(max_depth, water)

        risk_score = self._risk_score(max_depth, grid.vulnerability)
        flooded_cells = int(np.sum(max_depth > 0.12))
        water_volume = float(np.sum(water) * cell_area)
        return SimulationOutput(
            water_depth_m=water,
            max_water_depth_m=max_depth,
            risk_score=risk_score,
            water_volume_m3=water_volume,
            flooded_cells=flooded_cells,
        )

    def _redistribute(self, water: np.ndarray, elevation: np.ndarray, building_mask: np.ndarray) -> np.ndarray:
        head = elevation + water
        new_water = water.copy()

        for axis, shift in ((0, 1), (1, 1)):
            head_next = np.roll(head, -shift, axis=axis)
            water_next = np.roll(new_water, -shift, axis=axis)
            building_next = np.roll(building_mask, -shift, axis=axis)

            valid_edge = np.ones_like(building_mask, dtype=bool)
            if axis == 0:
                valid_edge[-1, :] = False
            else:
                valid_edge[:, -1] = False
            valid_edge &= ~building_mask & ~building_next

            delta = head - head_next
            transfer = np.where(valid_edge, np.clip(delta * self.lateral_flow_rate, 0.0, None), 0.0)
            transfer = np.minimum(transfer, new_water)

            new_water -= transfer
            water_next += transfer
            new_water = np.where(valid_edge, new_water, new_water)

            rolled_back = np.roll(water_next, shift, axis=axis)
            update_mask = np.roll(valid_edge, shift, axis=axis)
            new_water = np.where(update_mask, rolled_back, new_water)

        return np.maximum(new_water, 0.0)

    @staticmethod
    def _risk_score(max_depth: np.ndarray, vulnerability: np.ndarray) -> np.ndarray:
        capped_depth = np.clip(max_depth / 0.6, 0.0, 1.0)
        return np.clip(0.7 * capped_depth + 0.3 * vulnerability, 0.0, 1.0)
