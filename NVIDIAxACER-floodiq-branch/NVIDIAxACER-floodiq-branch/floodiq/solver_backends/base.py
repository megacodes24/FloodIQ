from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from floodiq.models import GridData, RainScenario, SimulationOutput


@dataclass(frozen=True)
class SolverMetadata:
    name: str
    engine: str
    device: str
    ready: bool
    details: str


class SolverBackend(Protocol):
    metadata: SolverMetadata

    def run(self, grid: GridData, scenario: RainScenario) -> SimulationOutput:
        ...
