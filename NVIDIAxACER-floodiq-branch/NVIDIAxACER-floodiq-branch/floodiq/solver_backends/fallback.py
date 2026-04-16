from __future__ import annotations

from floodiq.models import GridData, RainScenario, SimulationOutput
from floodiq.simulation import FloodSimulator

from .base import SolverMetadata


class FallbackSolverBackend:
    def __init__(self) -> None:
        self.simulator = FloodSimulator()
        self.metadata = SolverMetadata(
            name="fallback",
            engine="numpy shallow-water-inspired solver",
            device="cpu",
            ready=True,
            details="Deterministic local solver used for demos, validation, and surrogate supervision.",
        )

    def run(self, grid: GridData, scenario: RainScenario) -> SimulationOutput:
        return self.simulator.run(grid, scenario)
