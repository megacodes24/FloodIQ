from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .models import GridData, RainScenario
from .settings import StudyArea


IDA_START = pd.Timestamp("2021-08-31T00:00:00Z")
IDA_END = pd.Timestamp("2021-09-05T00:00:00Z")
IDA_SCENARIO = RainScenario(
    name="Hurricane Ida replay",
    rainfall_inches_per_hour=3.15,
    duration_hours=1.0,
)


@dataclass(frozen=True)
class EvaluationCaches:
    validation: dict[str, Any]
    benchmark: dict[str, Any]


def build_evaluation_summary(
    solver: Any,
    grid: GridData,
    area: StudyArea,
    sewer_complaints: pd.DataFrame | None,
    analysis_layers: dict[str, np.ndarray] | None,
    build_duration_ms: float | None,
) -> EvaluationCaches:
    return EvaluationCaches(
        validation=_validation_summary(solver, grid, area, sewer_complaints, analysis_layers),
        benchmark=_benchmark_summary(solver, grid, build_duration_ms),
    )


def _validation_summary(
    solver: Any,
    grid: GridData,
    area: StudyArea,
    sewer_complaints: pd.DataFrame | None,
    analysis_layers: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    if sewer_complaints is None or sewer_complaints.empty or "created_date" not in sewer_complaints:
        return {
            "status": "unavailable",
            "headline": "Historical validation unavailable.",
            "detail": "No complaint history is loaded for this study area.",
            "windows": [],
        }

    segment_labels = (analysis_layers or {}).get("segment_labels")
    if segment_labels is None:
        return {
            "status": "unavailable",
            "headline": "Street-segment validation unavailable.",
            "detail": "No street segment labels are loaded for this study area.",
            "windows": [],
        }

    solver_result = solver.run(grid, IDA_SCENARIO)
    predicted_scores = _segment_risk_scores(segment_labels, solver_result.risk_score)
    predicted_neighborhood_scores = _neighborhood_risk_scores(predicted_scores, area)
    if not predicted_scores:
        return {
            "status": "unavailable",
            "headline": "Validation could not compute segment scores.",
            "detail": "No predicted street segments were available for this study area.",
            "windows": [],
        }

    max_date = sewer_complaints["created_date"].max()
    windows = [
        ("Hurricane Ida event window", IDA_START, IDA_END),
        ("Recent 90 days", max_date - pd.Timedelta(days=90), max_date + pd.Timedelta(seconds=1)),
        ("Rolling 12 months", max_date - pd.Timedelta(days=365), max_date + pd.Timedelta(seconds=1)),
    ]

    window_summaries = [
        _window_summary(
            label=label,
            start=start,
            end=end,
            complaints=sewer_complaints,
            area=area,
            segment_labels=segment_labels,
            predicted_scores=predicted_scores,
            predicted_neighborhood_scores=predicted_neighborhood_scores,
        )
        for label, start, end in windows
    ]
    ready_windows = [window for window in window_summaries if window["status"] == "ready"]
    if not ready_windows:
        return {
            "status": "unavailable",
            "headline": "Validation windows did not contain usable complaint history.",
            "detail": "The loaded complaint history did not map cleanly onto the current study area.",
            "windows": window_summaries,
        }

    best_segment_window = max(
        ready_windows,
        key=lambda item: (item["complaint_capture_ratio"], item["hotspot_overlap_ratio"]),
    )
    best_neighborhood_window = max(
        ready_windows,
        key=lambda item: (item["neighborhood_capture_ratio"], item["neighborhood_overlap_ratio"]),
    )

    return {
        "status": "ready",
        "headline": (
            f"Across {best_segment_window['event'].lower()}, FloodIQ's top "
            f"{best_segment_window['complaint_capture_top_n']} predicted street segments capture "
            f"{round(best_segment_window['complaint_capture_ratio'] * 100)}% of complaint reports."
        ),
        "detail": (
            f"Neighborhood-level alignment is stronger: in {best_neighborhood_window['event'].lower()}, "
            f"the top {best_neighborhood_window['neighborhood_capture_top_n']} predicted neighborhoods capture "
            f"{round(best_neighborhood_window['neighborhood_capture_ratio'] * 100)}% of complaint reports. "
            "Use the segment score as the stricter test and the neighborhood score as the operational planning signal."
        ),
        "best_segment_window": best_segment_window,
        "best_neighborhood_window": best_neighborhood_window,
        "windows": ready_windows,
    }


def _window_summary(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    complaints: pd.DataFrame,
    area: StudyArea,
    segment_labels: np.ndarray,
    predicted_scores: dict[str, float],
    predicted_neighborhood_scores: dict[str, float],
) -> dict[str, Any]:
    historical = complaints[(complaints["created_date"] >= start) & (complaints["created_date"] < end)].copy()
    if historical.empty:
        return {
            "status": "unavailable",
            "event": label,
            "detail": "No complaint history in this window.",
        }

    segment_hits = _complaints_to_segments(historical, segment_labels, area)
    observed_ranked = sorted(segment_hits.items(), key=lambda item: item[1], reverse=True)
    observed_nonzero = [item for item in observed_ranked if item[1] > 0]
    if not observed_nonzero:
        return {
            "status": "unavailable",
            "event": label,
            "detail": "Complaints in this window did not map onto active street segments.",
        }

    top_k = min(5, len(observed_nonzero))
    capture_top_n = max(top_k, int(np.ceil(len(predicted_scores) * 0.1)))
    predicted_ranked = sorted(predicted_scores.items(), key=lambda item: item[1], reverse=True)
    predicted_hotspots = predicted_ranked[:top_k]
    predicted_capture = predicted_ranked[:capture_top_n]
    observed_top = observed_nonzero[:top_k]
    observed_ids = {segment_name for segment_name, _ in observed_top}
    predicted_ids = {segment_name for segment_name, _ in predicted_hotspots}
    overlap_ids = observed_ids & predicted_ids
    total_historical_complaints = sum(count for _, count in observed_nonzero)
    captured_complaints = sum(segment_hits.get(segment_name, 0) for segment_name, _ in predicted_capture)

    neighborhood_hits = _segment_counts_to_neighborhood_counts(segment_hits, area)
    observed_neighborhoods = sorted(neighborhood_hits.items(), key=lambda item: item[1], reverse=True)
    neighborhood_top_k = min(3, len(observed_neighborhoods))
    neighborhood_capture_top_n = max(1, min(3, len(predicted_neighborhood_scores)))
    predicted_neighborhood_ranked = sorted(predicted_neighborhood_scores.items(), key=lambda item: item[1], reverse=True)
    observed_neighborhood_ids = {name for name, _ in observed_neighborhoods[:neighborhood_top_k]}
    predicted_neighborhood_ids = {name for name, _ in predicted_neighborhood_ranked[:neighborhood_top_k]}
    neighborhood_overlap_ids = observed_neighborhood_ids & predicted_neighborhood_ids
    captured_neighborhood_complaints = sum(
        neighborhood_hits.get(name, 0) for name, _ in predicted_neighborhood_ranked[:neighborhood_capture_top_n]
    )

    return {
        "status": "ready",
        "event": label,
        "historical_rows": int(len(historical)),
        "historical_complaint_count": int(total_historical_complaints),
        "hotspot_overlap_ratio": round(len(overlap_ids) / top_k if top_k else 0.0, 3),
        "hotspot_overlap_count": int(len(overlap_ids)),
        "hotspot_overlap_top_k": int(top_k),
        "complaint_capture_ratio": round(captured_complaints / total_historical_complaints if total_historical_complaints else 0.0, 3),
        "complaint_capture_count": int(captured_complaints),
        "complaint_capture_top_n": int(capture_top_n),
        "neighborhood_overlap_ratio": round(
            len(neighborhood_overlap_ids) / neighborhood_top_k if neighborhood_top_k else 0.0,
            3,
        ),
        "neighborhood_overlap_count": int(len(neighborhood_overlap_ids)),
        "neighborhood_overlap_top_k": int(neighborhood_top_k),
        "neighborhood_capture_ratio": round(
            captured_neighborhood_complaints / total_historical_complaints if total_historical_complaints else 0.0,
            3,
        ),
        "neighborhood_capture_count": int(captured_neighborhood_complaints),
        "neighborhood_capture_top_n": int(neighborhood_capture_top_n),
        "observed_examples": [
            {"name": segment_name, "complaints": int(count)}
            for segment_name, count in observed_top[:3]
        ],
        "predicted_examples": [
            {"name": segment_name, "risk_score": round(float(score), 3)}
            for segment_name, score in predicted_hotspots[:3]
        ],
        "observed_neighborhood_examples": [
            {"name": name, "complaints": int(count)}
            for name, count in observed_neighborhoods[:3]
        ],
        "predicted_neighborhood_examples": [
            {"name": name, "risk_score": round(float(score), 3)}
            for name, score in predicted_neighborhood_ranked[:3]
        ],
    }


def _benchmark_summary(solver: Any, grid: GridData, build_duration_ms: float | None) -> dict[str, Any]:
    warm_scenario = RainScenario(
        name="Benchmark storm",
        rainfall_inches_per_hour=3.15,
        duration_hours=1.0,
    )
    timings_ms: list[float] = []
    for _ in range(3):
        started = perf_counter()
        solver.run(grid, warm_scenario)
        timings_ms.append((perf_counter() - started) * 1000.0)

    inference_ms = float(np.mean(timings_ms))
    return {
        "status": "ready",
        "grid_build_ms": round(build_duration_ms or 0.0, 1),
        "mean_inference_ms": round(inference_ms, 1),
        "p95_inference_ms": round(float(np.percentile(timings_ms, 95)), 1),
        "runs_profiled": len(timings_ms),
        "headline": (
            f"{solver.metadata.name} averages {round(inference_ms, 1)} ms per Lower Manhattan storm scenario."
        ),
        "detail": (
            f"Benchmark runs the active solver three times on a {grid.elevation.shape[0]}x{grid.elevation.shape[1]} "
            "Lower Manhattan grid after local NYC data ingestion and preprocessing."
        ),
        "stack": {
            "engine": solver.metadata.engine,
            "device": solver.metadata.device,
            "ready": solver.metadata.ready,
        },
        "checkpoint": _checkpoint_metadata(solver),
    }


def _complaints_to_segments(
    complaints: pd.DataFrame,
    segment_labels: np.ndarray,
    area: StudyArea,
) -> dict[str, int]:
    rows, cols = segment_labels.shape
    counts: dict[str, int] = {}
    lat_span = area.lat_max - area.lat_min
    lon_span = area.lon_max - area.lon_min
    for record in complaints[["latitude", "longitude"]].dropna().to_dict(orient="records"):
        col = int(np.clip(((record["longitude"] - area.lon_min) / lon_span) * cols, 0, cols - 1))
        row = int(np.clip(((area.lat_max - record["latitude"]) / lat_span) * rows, 0, rows - 1))
        segment_name = str(segment_labels[row, col]).strip()
        if segment_name:
            counts[segment_name] = counts.get(segment_name, 0) + 1
    return counts


def _segment_risk_scores(segment_labels: np.ndarray, risk_score: np.ndarray) -> dict[str, float]:
    scores: dict[str, float] = {}
    for segment_name in np.unique(segment_labels):
        name = str(segment_name).strip()
        if not name:
            continue
        mask = segment_labels == segment_name
        scores[name] = float(np.mean(risk_score[mask]))
    return scores


def _segment_counts_to_neighborhood_counts(segment_counts: dict[str, int], area: StudyArea) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment_name, count in segment_counts.items():
        neighborhood = _infer_neighborhood_from_segment(segment_name, area)
        counts[neighborhood] = counts.get(neighborhood, 0) + count
    return counts


def _neighborhood_risk_scores(segment_scores: dict[str, float], area: StudyArea) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for segment_name, score in segment_scores.items():
        neighborhood = _infer_neighborhood_from_segment(segment_name, area)
        buckets.setdefault(neighborhood, []).append(score)
    return {
        name: float(np.mean(values))
        for name, values in buckets.items()
    }


def _infer_neighborhood_from_segment(segment_name: str, area: StudyArea) -> str:
    if area.slug != "lower_manhattan":
        return area.name
    upper = segment_name.upper()
    if any(token in upper for token in ("VARICK", "WASHINGTON", "HUDSON", "CANAL", "WEST", "DESBROSSES")):
        return "Hudson Square / West Side"
    if any(token in upper for token in ("BOWERY", "DELANCEY", "RIVINGTON", "HOUSTON", "ESSEX", "FORSYTH")):
        return "Lower East Side / Nolita Edge"
    if any(token in upper for token in ("WATER", "FRONT", "PEARL", "FULTON", "CATHERINE", "SOUTH", "LIBERTY", "BEAVER")):
        return "Financial District / Seaport"
    return "Civic Center / Chinatown"


def _checkpoint_metadata(solver: Any) -> dict[str, Any] | None:
    checkpoint_path = getattr(solver, "checkpoint_path", None)
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        import torch  # type: ignore

        payload = torch.load(path, map_location="cpu")
        model_state = payload.get("model_state") or {}
        param_count = int(sum(int(tensor.numel()) for tensor in model_state.values())) if model_state else None
        return {
            "path": str(path),
            "study_area": payload.get("study_area"),
            "samples": payload.get("samples"),
            "epochs": payload.get("epochs"),
            "grid_shape": payload.get("grid_shape"),
            "complaint_guided": payload.get("complaint_guided"),
            "param_count": param_count,
        }
    except Exception:
        return {"path": str(path)}
