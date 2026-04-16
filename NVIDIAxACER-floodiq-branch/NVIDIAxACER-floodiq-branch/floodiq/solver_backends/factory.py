from __future__ import annotations

import os

from .fallback import FallbackSolverBackend
from .physicsnemo_backend import PhysicsNeMoSurrogateBackend


def build_solver_backend() -> FallbackSolverBackend | PhysicsNeMoSurrogateBackend:
    selected = os.environ.get("FLOODIQ_SOLVER", "auto").strip().lower()
    if selected in {"physicsnemo", "auto"}:
        backend = PhysicsNeMoSurrogateBackend()
        if backend.metadata.ready:
            return backend
    return FallbackSolverBackend()
