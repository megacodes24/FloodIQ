from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from .data import load_demo_grid
from .evaluation import build_evaluation_summary
from .noaa import NOAAClient
from .models import RainScenario
from .nyc_open_data import NYCOpenDataClient
from .providers import detect_runtime_capabilities
from .real_grid import build_real_grid
from .settings import DEFAULT_STUDY_AREA, STUDY_AREAS, StudyArea, get_study_area
from .solver_backends import build_solver_backend


class FloodIQService:
    def __init__(self, use_live_data: bool = True, study_area: StudyArea | None = None) -> None:
        self.capabilities = detect_runtime_capabilities()
        self.study_area = study_area or DEFAULT_STUDY_AREA
        self.data_mode = "synthetic"
        self.grid_metadata: dict[str, Any] = {"mode": "synthetic"}
        self.noaa_forecast: dict[str, Any] | None = None
        self.sewer_complaints = None
        self.analysis_layers: dict[str, np.ndarray] = {}
        self.grid_build_duration_ms: float | None = None
        self._evaluation_cache: dict[str, Any] | None = None
        self.grid = self._load_grid(use_live_data=use_live_data)
        self.solver = build_solver_backend()

    def scenario_catalog(self) -> list[dict[str, Any]]:
        scenarios = [
            {
                "name": "Summer cloudburst",
                "rainfall_inches_per_hour": 1.6,
                "duration_hours": 1.0,
                "description": "Short burst that overwhelms low-capacity drains.",
            },
            {
                "name": "Hurricane Ida replay",
                "rainfall_inches_per_hour": 3.15,
                "duration_hours": 1.0,
                "description": "High-intensity rainfall tuned to the Ida demo narrative.",
            },
            {
                "name": "Slow nor'easter",
                "rainfall_inches_per_hour": 0.9,
                "duration_hours": 4.0,
                "description": "Long-duration storm that tests drainage accumulation.",
            },
        ]
        if self.noaa_forecast is not None:
            scenarios.insert(0, self.noaa_forecast)
        return scenarios

    def default_scenario(self) -> dict[str, Any]:
        if self.noaa_forecast is not None:
            return self.noaa_forecast
        return self.scenario_catalog()[0]

    def run_scenario(
        self,
        rainfall_inches_per_hour: float | None = None,
        duration_hours: float | None = None,
        name: str = "Custom scenario",
    ) -> dict[str, Any]:
        default_scenario = self.default_scenario()
        rainfall_inches_per_hour = float(
            rainfall_inches_per_hour if rainfall_inches_per_hour is not None else default_scenario["rainfall_inches_per_hour"]
        )
        duration_hours = float(duration_hours if duration_hours is not None else default_scenario["duration_hours"])
        if name == "Custom scenario" and default_scenario["name"] == "Live NOAA forecast":
            name = "Forecast-driven flood run"
        scenario = RainScenario(
            name=name,
            rainfall_inches_per_hour=rainfall_inches_per_hour,
            duration_hours=duration_hours,
        )
        result = self.solver.run(self.grid, scenario)
        blocks = self._summarize_blocks(result.max_water_depth_m, result.risk_score)
        neighborhoods = self._summarize_neighborhoods(blocks)
        explanation = self._top_risk_explanation(blocks)

        highest_risk = max(blocks, key=lambda block: block["risk_score"])
        return {
            "scenario": asdict(scenario),
            "forecast_context": self.noaa_forecast,
            "summary": {
                "flooded_cells": result.flooded_cells,
                "water_volume_m3": round(result.water_volume_m3, 2),
                "peak_depth_m": round(float(np.max(result.max_water_depth_m)), 3),
                "top_block": highest_risk["name"],
                "top_block_depth_m": highest_risk["max_depth_m"],
                "top_block_risk": highest_risk["risk_score"],
            },
            "grid": {
                "rows": int(result.max_water_depth_m.shape[0]),
                "cols": int(result.max_water_depth_m.shape[1]),
                "max_water_depth_m": np.round(result.max_water_depth_m, 3).tolist(),
                "building_mask": self.grid.building_mask.astype(int).tolist(),
                "vulnerability": np.round(self.grid.vulnerability, 2).tolist(),
            },
            "map": {
                "center": {
                    "lat": round(self.study_area.center[0], 6),
                    "lon": round(self.study_area.center[1], 6),
                },
                "bounds": {
                    "lat_min": self.study_area.lat_min,
                    "lat_max": self.study_area.lat_max,
                    "lon_min": self.study_area.lon_min,
                    "lon_max": self.study_area.lon_max,
                },
                "risk_blocks": self._map_blocks(blocks),
                "risk_segments": self._map_segments(result.max_water_depth_m, result.risk_score),
                "neighborhoods": neighborhoods,
            },
            "blocks": blocks,
            "neighborhoods": neighborhoods,
            "explanation": explanation,
            "alerts": self._alerts(blocks),
            "recommended_actions": self._recommended_actions(blocks),
            "data_mode": self.data_mode,
            "solver": {
                "name": self.solver.metadata.name,
                "engine": self.solver.metadata.engine,
                "device": self.solver.metadata.device,
                "ready": self.solver.metadata.ready,
                "details": self.solver.metadata.details,
            },
            "data_sources": self.grid_metadata,
            "nvidia_story": {
                "today": (
                    "This machine is using the CPU path. On the Acer GN100, the same preprocessing pipeline "
                    "can use cuDF and the solver backend can be upgraded to PhysicsNeMo/Modulus."
                ),
                "upgrade_path": [
                    "Swap the simulation kernel with NVIDIA PhysicsNeMo/Modulus for full SWE PDE solving.",
                    "Use RAPIDS cuDF and cuSpatial to preprocess LiDAR, drains, and complaint joins on GPU.",
                    "Attach a local NIM endpoint to turn block-level predictions into plain-English emergency briefings.",
                ],
            },
        }

    def baseline_payload(self) -> dict[str, Any]:
        evaluation = self._evaluation_payload()
        checkpoint = evaluation.get("benchmark", {}).get("checkpoint") or {}
        return {
            "title": "FloodIQ",
            "subtitle": "Forecast-driven flood forecasting for NYC's clogging-prone streets.",
            "scenarios": self.scenario_catalog(),
            "default_scenario": self.default_scenario(),
            "study_area": {
                "name": self.study_area.name,
                "slug": self.study_area.slug,
            },
            "weather": {
                "forecast_available": self.noaa_forecast is not None,
                "active_forecast": self.noaa_forecast,
                "status": self.grid_metadata.get(
                    "forecast_status",
                    "NOAA forecast ready" if self.noaa_forecast else "No rain forecast currently available",
                ),
            },
            "study_areas": [
                {"name": area.name, "slug": area.slug}
                for area in STUDY_AREAS.values()
            ],
            "data_mode": self.data_mode,
            "capabilities": {
                "cudf": self.capabilities.cudf,
                "cuspatial": self.capabilities.cuspatial,
                "physicsnemo": self.capabilities.physicsnemo,
                "torch_cuda": self.capabilities.torch_cuda,
            },
            "solver": {
                "name": self.solver.metadata.name,
                "ready": self.solver.metadata.ready,
            },
            "demo_stack": {
                "location": "Acer GN100 local runtime",
                "model": "NVIDIA PhysicsNeMo FNO",
                "params": checkpoint.get("param_count"),
                "samples": checkpoint.get("samples"),
                "grid_shape": checkpoint.get("grid_shape"),
                "complaint_guided": checkpoint.get("complaint_guided"),
            },
            "evaluation": evaluation,
            "about": [
                "Pull the latest forecast before the rain arrives.",
                "See which streets and neighborhoods accumulate water first.",
                "Prioritize drains, roads, and vulnerable neighborhoods before impact.",
            ],
        }

    def _load_grid(self, use_live_data: bool) -> Any:
        load_started = perf_counter()
        if not use_live_data:
            self.grid_metadata = {
                "mode": "synthetic",
                "fallback_reason": "Live data disabled for this run.",
            }
            self.grid_build_duration_ms = (perf_counter() - load_started) * 1000.0
            return load_demo_grid()
        try:
            nyc = NYCOpenDataClient()
            bundle = nyc.fetch_bundle(self.study_area)
            self.sewer_complaints = bundle.sewer_complaints
            real_grid = build_real_grid(
                bundle.sewer_complaints,
                bundle.elevation_points,
                bundle.street_centerlines,
                bundle.sidewalk_polygons,
                bundle.parking_lot_polygons,
                bundle.catch_basins,
                bundle.outfalls,
                self.study_area,
                self.capabilities,
            )
            self.analysis_layers = real_grid.layers
            self.data_mode = "real"
            self.grid_metadata = {
                **bundle.metadata,
                **real_grid.metadata,
            }
            try:
                lat, lon = self.study_area.center
                forecast = NOAAClient().fetch_quantitative_precipitation(lat, lon)
                if forecast is not None:
                    self.noaa_forecast = {
                        "name": forecast.name,
                        "rainfall_inches_per_hour": forecast.rainfall_inches_per_hour,
                        "duration_hours": forecast.duration_hours,
                        "valid_time": forecast.valid_time,
                        "start_time_iso": forecast.start_time_iso,
                        "start_time_label": forecast.start_time_label,
                        "hours_until_start": forecast.hours_until_start,
                        "source_summary": forecast.source_summary,
                        "precipitation_probability": forecast.precipitation_probability,
                        "description": f"{forecast.source_summary} ({forecast.source_url}).",
                    }
            except Exception as forecast_exc:
                self.grid_metadata["forecast_status"] = f"NOAA unavailable: {forecast_exc}"
            self.grid_build_duration_ms = (perf_counter() - load_started) * 1000.0
            return real_grid.grid
        except Exception as exc:
            self.grid_metadata = {
                "mode": "synthetic",
                "fallback_reason": str(exc),
            }
            self.grid_build_duration_ms = (perf_counter() - load_started) * 1000.0
            return load_demo_grid()

    def _summarize_blocks(self, depth: np.ndarray, risk: np.ndarray) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block_id, name in enumerate(self.grid.block_names):
            mask = self.grid.block_ids == block_id
            max_depth = float(np.max(depth[mask]))
            avg_risk = float(np.mean(risk[mask]))
            avg_vulnerability = float(np.mean(self.grid.vulnerability[mask]))
            row_idx, col_idx = np.argwhere(mask)[0]
            lat, lon = self._cell_to_lat_lon(int(row_idx), int(col_idx), depth.shape[0], depth.shape[1])
            blocks.append(
                {
                    "id": block_id,
                    "name": name,
                    "max_depth_m": round(max_depth, 3),
                    "risk_score": round(avg_risk, 3),
                    "vulnerability": round(avg_vulnerability, 3),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                }
            )
        blocks.sort(key=lambda item: (item["risk_score"], item["max_depth_m"]), reverse=True)
        return blocks

    @staticmethod
    def _alerts(blocks: list[dict[str, Any]]) -> list[str]:
        top = blocks[:3]
        return [
            (
                f"{block['name']} is projected to reach {block['max_depth_m']}m of standing water "
                "during the incoming rain window."
            )
            for block in top
        ]

    def _map_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for block in blocks[:18]:
            mapped.append(
                {
                    "name": block["name"],
                    "lat": block["lat"],
                    "lon": block["lon"],
                    "risk_score": block["risk_score"],
                    "max_depth_m": block["max_depth_m"],
                    "radius_m": int(80 + (block["risk_score"] * 220)),
                }
            )
        return mapped

    def _map_segments(self, depth: np.ndarray, risk: np.ndarray) -> list[dict[str, Any]]:
        street_segments = self.analysis_layers.get("street_segments")
        if not street_segments:
            return []

        segments: list[dict[str, Any]] = []
        for segment in street_segments:
            segment_name = str(segment.get("name") or "").strip()
            coordinates = segment.get("coordinates") or []
            cells = segment.get("cells") or []
            if not segment_name or len(coordinates) < 2 or not cells:
                continue
            unique_cells = {(int(row), int(col)) for row, col in cells}
            rows = np.asarray([row for row, _ in unique_cells], dtype=int)
            cols = np.asarray([col for _, col in unique_cells], dtype=int)
            segment_depth = float(np.max(depth[rows, cols]))
            segment_risk = float(np.mean(risk[rows, cols]))
            if segment_risk < 0.28 and segment_depth < 0.05:
                continue
            centroid_lat = float(np.mean([point[0] for point in coordinates]))
            centroid_lon = float(np.mean([point[1] for point in coordinates]))
            segments.append(
                {
                    "name": segment_name,
                    "risk_score": round(segment_risk, 3),
                    "max_depth_m": round(segment_depth, 3),
                    "coordinates": [[round(float(lat), 6), round(float(lon), 6)] for lat, lon in coordinates],
                    "lat": round(centroid_lat, 6),
                    "lon": round(centroid_lon, 6),
                    "neighborhood": self._infer_neighborhood(centroid_lat, centroid_lon),
                }
            )

        segments.sort(key=lambda item: (item["risk_score"], item["max_depth_m"]), reverse=True)
        return segments[:52]

    def _cell_to_lat_lon(self, row: int, col: int, rows: int, cols: int) -> tuple[float, float]:
        lat = self.study_area.lat_max - ((row + 0.5) / rows) * (self.study_area.lat_max - self.study_area.lat_min)
        lon = self.study_area.lon_min + ((col + 0.5) / cols) * (self.study_area.lon_max - self.study_area.lon_min)
        return lat, lon

    def _recommended_actions(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        zero_layer = np.zeros_like(self.grid.block_ids, dtype=float)
        recent_counts = self.analysis_layers.get("recent_counts", zero_layer)
        catch_basin_counts = self.analysis_layers.get("catch_basin_counts", zero_layer)
        outfall_counts = self.analysis_layers.get("outfall_counts", zero_layer)
        hard_surface = self.analysis_layers.get("hard_surface", zero_layer)

        for rank, block in enumerate(blocks[:3], start=1):
            mask = self.grid.block_ids == block["id"]
            recent_pressure = int(np.sum(recent_counts[mask]))
            nearby_catch_basins = int(np.sum(catch_basin_counts[mask]))
            nearby_outfalls = int(np.sum(outfall_counts[mask]))
            paved_share = float(np.mean(hard_surface[mask]))

            if nearby_catch_basins <= 1:
                title = f"Priority {rank}: inspect drainage assets near {block['name']}"
                detail = (
                    f"{block['name']} is high risk with only {nearby_catch_basins} mapped catch basin cells nearby. "
                    "Dispatch a crew to clear curb inlets and remove blockage before rainfall peaks."
                )
            elif nearby_outfalls > 0:
                title = f"Priority {rank}: verify outfall path at {block['name']}"
                detail = (
                    f"{block['name']} drains toward {nearby_outfalls} mapped outfall cells. "
                    "Pre-stage waterfront pumping and confirm downstream discharge is unobstructed."
                )
            else:
                title = f"Priority {rank}: pre-stage crews at {block['name']}"
                detail = (
                    f"{block['name']} combines elevated modeled depth with {recent_pressure} recent complaint hits "
                    f"and {paved_share:.0%} hard-surface coverage. Prepare traffic control and targeted drain inspection."
                )

            actions.append(
                {
                    "title": title,
                    "detail": detail,
                    "block": block["name"],
                    "neighborhood": self._infer_neighborhood(block["lat"], block["lon"]),
                    "risk_score": block["risk_score"],
                }
            )
        return actions

    def _top_risk_explanation(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        if not blocks:
            return {
                "headline": "No high-risk streets identified yet.",
                "drivers": [],
                "location": self.study_area.name,
            }
        block = blocks[0]
        zero_layer = np.zeros_like(self.grid.block_ids, dtype=float)
        recent_counts = self.analysis_layers.get("recent_counts", zero_layer)
        catch_basin_counts = self.analysis_layers.get("catch_basin_counts", zero_layer)
        outfall_counts = self.analysis_layers.get("outfall_counts", zero_layer)
        hard_surface = self.analysis_layers.get("hard_surface", zero_layer)
        mask = self.grid.block_ids == block["id"]
        recent_pressure = int(np.sum(recent_counts[mask]))
        nearby_catch_basins = int(np.sum(catch_basin_counts[mask]))
        nearby_outfalls = int(np.sum(outfall_counts[mask]))
        paved_share = float(np.mean(hard_surface[mask]))

        drivers: list[str] = []
        if recent_pressure > 20:
            drivers.append(f"{recent_pressure} recent complaint hits indicate repeated drainage trouble.")
        if paved_share > 0.75:
            drivers.append(f"{paved_share:.0%} hard-surface coverage limits infiltration.")
        if nearby_catch_basins <= 1:
            drivers.append("Few mapped catch basin cells nearby increase local backup risk.")
        elif nearby_outfalls > 0:
            drivers.append(f"{nearby_outfalls} mapped outfall cells suggest concentrated downstream flow pressure.")
        if not drivers:
            drivers.append("Modeled depth and vulnerability combine to make this corridor the top current risk.")

        return {
            "location": block["name"],
            "neighborhood": self._infer_neighborhood(block["lat"], block["lon"]),
            "headline": (
                f"{block['name']} is the first street to inspect before the forecast window."
            ),
            "drivers": drivers[:3],
        }

    def _summarize_neighborhoods(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_neighborhood: dict[str, dict[str, Any]] = {}
        for block in blocks[:18]:
            neighborhood = self._infer_neighborhood(block["lat"], block["lon"])
            bucket = by_neighborhood.setdefault(
                neighborhood,
                {
                    "name": neighborhood,
                    "risk_scores": [],
                    "peak_depth_m": 0.0,
                    "locations": [],
                },
            )
            bucket["risk_scores"].append(block["risk_score"])
            bucket["peak_depth_m"] = max(bucket["peak_depth_m"], block["max_depth_m"])
            if len(bucket["locations"]) < 3:
                bucket["locations"].append(block["name"])

        summaries = [
            {
                "name": name,
                "risk_score": round(float(np.mean(payload["risk_scores"])), 3),
                "peak_depth_m": round(float(payload["peak_depth_m"]), 3),
                "locations": payload["locations"],
            }
            for name, payload in by_neighborhood.items()
        ]
        summaries.sort(key=lambda item: (item["risk_score"], item["peak_depth_m"]), reverse=True)
        return summaries[:4]

    def _infer_neighborhood(self, lat: float, lon: float) -> str:
        if self.study_area.slug != "lower_manhattan":
            return self.study_area.name
        if lat > 40.7205 and lon < -73.997:
            return "Hudson Square / SoHo Edge"
        if lat > 40.7205:
            return "Lower East Side / Nolita Edge"
        if lon < -74.008:
            return "Battery Park / West Side"
        if lat < 40.7095:
            return "Financial District / Seaport"
        return "Civic Center / Chinatown"

    def _evaluation_payload(self) -> dict[str, Any]:
        if self._evaluation_cache is None:
            caches = build_evaluation_summary(
                solver=self.solver,
                grid=self.grid,
                area=self.study_area,
                sewer_complaints=self.sewer_complaints,
                analysis_layers=self.analysis_layers,
                build_duration_ms=self.grid_build_duration_ms,
            )
            self._evaluation_cache = {
                "validation": caches.validation,
                "benchmark": caches.benchmark,
            }
        return self._evaluation_cache


def build_service_for_slug(study_area_slug: str | None, use_live_data: bool = True) -> FloodIQService:
    return FloodIQService(use_live_data=use_live_data, study_area=get_study_area(study_area_slug))
