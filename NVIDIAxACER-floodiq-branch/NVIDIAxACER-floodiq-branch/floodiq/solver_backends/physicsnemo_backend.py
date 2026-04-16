from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from floodiq.models import GridData, RainScenario, SimulationOutput
from floodiq.settings import PROJECT_ROOT

from .base import SolverMetadata
from .features import scenario_feature_stack
from .physicsnemo_compat import ensure_physicsnemo_torch_compat


DEFAULT_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "physicsnemo_lower_manhattan_heavy.pt",
    PROJECT_ROOT / "artifacts" / "physicsnemo_lower_manhattan.pt",
    PROJECT_ROOT / "artifacts" / "physicsnemo_surrogate.pt",
]


class PhysicsNeMoSurrogateBackend:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        env_checkpoint = os.environ.get("FLOODIQ_PHYSICSNEMO_CHECKPOINT")
        self.checkpoint_path = checkpoint_path or self._resolve_checkpoint_path(env_checkpoint)
        self._model: Any | None = None
        self._device = "cuda"
        self._load_error: str | None = None
        self._target_scale: float = 1.0
        self._target_transform: str = "identity"
        self.metadata = self._bootstrap_metadata()

    def run(self, grid: GridData, scenario: RainScenario) -> SimulationOutput:
        if self._model is None:
            raise RuntimeError(self.metadata.details)

        import torch  # type: ignore

        features = scenario_feature_stack(grid, scenario)
        with torch.no_grad():
            tensor = torch.from_numpy(features).unsqueeze(0).to(self._device)
            prediction = self._model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()

        max_depth = self._decode_prediction(prediction)
        water_depth = max_depth.copy()
        capped_depth = np.clip(max_depth / 0.6, 0.0, 1.0)
        risk_score = np.clip(0.7 * capped_depth + 0.3 * grid.vulnerability, 0.0, 1.0)
        flooded_cells = int(np.sum(max_depth > 0.12))
        water_volume = float(np.sum(water_depth) * (20.0 ** 2))
        return SimulationOutput(
            water_depth_m=water_depth,
            max_water_depth_m=max_depth,
            risk_score=risk_score,
            water_volume_m3=water_volume,
            flooded_cells=flooded_cells,
        )

    def _bootstrap_metadata(self) -> SolverMetadata:
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return SolverMetadata(
                    name="physicsnemo",
                    engine="physicsnemo surrogate",
                    device="cpu",
                    ready=False,
                    details="PhysicsNeMo backend requires CUDA-enabled PyTorch.",
                )
            fno_cls = _load_fno_class()
            if not self.checkpoint_path.exists():
                return SolverMetadata(
                    name="physicsnemo",
                    engine="physicsnemo surrogate",
                    device="cuda",
                    ready=False,
                    details=(
                        f"Checkpoint not found at {self.checkpoint_path}. "
                        "Train the surrogate with `python -m floodiq.train_physicsnemo_surrogate` first."
                    ),
                )
            state = torch.load(self.checkpoint_path, map_location=self._device)
            architecture = _checkpoint_architecture(state)
            self._model = fno_cls(
                **architecture,
            ).to(self._device)
            self._model.load_state_dict(state["model_state"])
            self._target_scale = float(state.get("target_scale", 1.0))
            self._target_transform = str(state.get("target_transform", "identity"))
            self._model.eval()
            return SolverMetadata(
                name="physicsnemo",
                engine="physicsnemo FNO surrogate",
                device="cuda",
                ready=True,
                details=f"Loaded PhysicsNeMo surrogate checkpoint from {self.checkpoint_path}.",
            )
        except Exception as exc:
            self._load_error = str(exc)
            return SolverMetadata(
                name="physicsnemo",
                engine="physicsnemo surrogate",
                device="cuda",
                ready=False,
                details=f"PhysicsNeMo backend unavailable: {exc}",
            )

    def _decode_prediction(self, prediction: np.ndarray) -> np.ndarray:
        clipped = np.clip(prediction, 0.0, None)
        if self._target_transform == "sqrt":
            return (clipped ** 2) * self._target_scale
        return clipped * self._target_scale

    @staticmethod
    def _resolve_checkpoint_path(env_checkpoint: str | None) -> Path:
        if env_checkpoint:
            return Path(env_checkpoint)
        for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
            if candidate.exists():
                return candidate
        return DEFAULT_CHECKPOINT_CANDIDATES[-1]


def _load_fno_class() -> Any:
    try:
        ensure_physicsnemo_torch_compat()
        from physicsnemo.models.fno.fno import FNO  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Unable to import PhysicsNeMo FNO. "
            "The current environment may have an incompatible torch/physicsnemo combination."
        ) from exc
    return FNO


def _checkpoint_architecture(state: dict[str, Any]) -> dict[str, Any]:
    architecture = state.get("architecture")
    if isinstance(architecture, dict):
        return {
            "in_channels": int(architecture.get("in_channels", 8)),
            "out_channels": int(architecture.get("out_channels", 1)),
            "dimension": int(architecture.get("dimension", 2)),
            "latent_channels": int(architecture.get("latent_channels", 24)),
            "num_fno_layers": int(architecture.get("num_fno_layers", 4)),
            "num_fno_modes": int(architecture.get("num_fno_modes", 12)),
            "padding": int(architecture.get("padding", 0)),
            "decoder_layers": int(architecture.get("decoder_layers", 2)),
            "decoder_layer_size": int(architecture.get("decoder_layer_size", 48)),
        }
    return {
        "in_channels": 8,
        "out_channels": 1,
        "dimension": 2,
        "latent_channels": 24,
        "num_fno_layers": 4,
        "num_fno_modes": 12,
        "padding": 0,
        "decoder_layers": 2,
        "decoder_layer_size": 48,
    }
