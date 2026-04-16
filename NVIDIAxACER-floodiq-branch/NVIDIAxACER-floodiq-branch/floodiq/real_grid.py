from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import GridData
from .providers import RuntimeCapabilities
from .settings import StudyArea


@dataclass(frozen=True)
class RealGridBuildResult:
    grid: GridData
    metadata: dict[str, Any]
    layers: dict[str, np.ndarray]


def build_real_grid(
    sewer_complaints: pd.DataFrame,
    elevation_points: pd.DataFrame,
    street_centerlines: pd.DataFrame,
    sidewalk_polygons: pd.DataFrame,
    parking_lot_polygons: pd.DataFrame,
    catch_basins: pd.DataFrame,
    outfalls: pd.DataFrame,
    area: StudyArea,
    capabilities: RuntimeCapabilities,
    size: int | None = None,
    block_span: int | None = None,
) -> RealGridBuildResult:
    size = size or (96 if area.slug == "lower_manhattan" else 48)
    block_span = block_span or max(8, size // 6)
    elevation_grid, elevation_density = _aggregate_points(
        elevation_points[["longitude", "latitude", "elevation"]].copy(),
        area,
        size,
        value_column="elevation",
        agg="mean",
        capabilities=capabilities,
    )
    complaint_counts, _ = _aggregate_points(
        sewer_complaints[["longitude", "latitude"]].copy(),
        area,
        size,
        value_column=None,
        agg="count",
        capabilities=capabilities,
    )

    recent_cutoff = sewer_complaints["created_date"].max() - pd.Timedelta(days=365) if not sewer_complaints.empty else None
    recent = sewer_complaints[sewer_complaints["created_date"] >= recent_cutoff] if recent_cutoff is not None else sewer_complaints
    recent_counts, _ = _aggregate_points(
        recent[["longitude", "latitude"]].copy(),
        area,
        size,
        value_column=None,
        agg="count",
        capabilities=capabilities,
    )
    catch_basin_counts, _ = _aggregate_points(
        catch_basins[["longitude", "latitude"]].copy(),
        area,
        size,
        value_column=None,
        agg="count",
        capabilities=capabilities,
    )
    outfall_counts, _ = _aggregate_points(
        outfalls[["longitude", "latitude"]].copy(),
        area,
        size,
        value_column=None,
        agg="count",
        capabilities=capabilities,
    )

    normalized_elevation = _normalize(elevation_grid, invert=True)
    normalized_density = _normalize(elevation_density)
    normalized_complaints = _normalize(complaint_counts)
    normalized_recent = _normalize(recent_counts)
    normalized_catch_basins = _normalize(catch_basin_counts)
    normalized_outfalls = _normalize(outfall_counts)
    street_mask, street_primary, street_secondary, street_segments = rasterize_centerlines(street_centerlines, area, size)
    sidewalk_mask = rasterize_polygons(sidewalk_polygons, area, size)
    parking_mask = rasterize_polygons(parking_lot_polygons, area, size)
    hard_surface = np.clip(street_mask + sidewalk_mask + parking_mask, 0.0, 1.0)

    synthetic_gradient = np.linspace(0.2, 1.0, size).reshape(size, 1)
    complaint_basin = 0.75 * normalized_complaints + 0.45 * normalized_recent
    elevation = (
        1.6
        + 1.9 * normalized_elevation
        + 0.7 * synthetic_gradient
        - 0.85 * complaint_basin
        - 0.22 * street_mask
        - 0.18 * normalized_outfalls
    )
    elevation = np.clip(elevation, 0.2, None)

    infiltration_rate = (
        1.2e-6
        - 9.0e-7 * normalized_density
        - 2.0e-7 * normalized_complaints
        - 5.0e-7 * hard_surface
        + 1.5e-7 * normalized_catch_basins
    )
    infiltration_rate = np.clip(infiltration_rate, 1.0e-7, 1.2e-6)

    drain_capacity = (
        1.6e-5
        - 1.4e-5 * normalized_complaints
        - 5.0e-6 * normalized_recent
        + 8.0e-6 * normalized_catch_basins
        + 1.0e-5 * normalized_outfalls
        + 5.0e-6 * street_mask
    )
    drain_capacity = np.clip(drain_capacity, 1.2e-6, 2.6e-5)

    building_mask = (normalized_density > np.quantile(normalized_density, 0.94)) & (street_mask < 0.2)
    vulnerability = np.clip(
        0.32
        + 0.38 * normalized_recent
        + 0.22 * normalized_complaints
        + 0.14 * hard_surface
        - 0.08 * normalized_catch_basins,
        0.0,
        1.0,
    )

    block_ids = np.full((size, size), -1, dtype=int)
    block_names: list[str] = []
    idx = 0
    blocks_per_axis = size // block_span
    for row in range(blocks_per_axis):
        for col in range(blocks_per_axis):
            r0 = row * block_span
            c0 = col * block_span
            block_ids[r0:r0 + block_span, c0:c0 + block_span] = idx
            block_names.append(_block_label_from_grid(street_primary, street_secondary, r0, c0, block_span, idx))
            idx += 1

    grid = GridData(
        elevation=elevation,
        infiltration_rate=infiltration_rate,
        drain_capacity=drain_capacity,
        building_mask=building_mask,
        vulnerability=vulnerability,
        block_ids=block_ids,
        block_names=block_names,
    )

    return RealGridBuildResult(
        grid=grid,
        metadata={
            "mode": "real",
            "study_area": area.name,
            "grid_size": size,
            "complaint_cells_with_events": int(np.sum(complaint_counts > 0)),
            "recent_complaint_cells_with_events": int(np.sum(recent_counts > 0)),
            "avg_elevation_point_density": round(float(np.mean(elevation_density)), 3),
            "street_cells": int(np.sum(street_mask > 0)),
            "sidewalk_cells": int(np.sum(sidewalk_mask > 0)),
            "parking_cells": int(np.sum(parking_mask > 0)),
            "catch_basin_cells": int(np.sum(catch_basin_counts > 0)),
            "outfall_cells": int(np.sum(outfall_counts > 0)),
            "gpu_ready": capabilities.cudf,
        },
        layers={
            "complaint_counts": complaint_counts,
            "recent_counts": recent_counts,
            "catch_basin_counts": catch_basin_counts,
            "outfall_counts": outfall_counts,
            "street_mask": street_mask,
            "street_primary": street_primary,
            "street_secondary": street_secondary,
            "segment_labels": _segment_label_grid(street_primary, street_secondary),
            "street_segments": street_segments,
            "sidewalk_mask": sidewalk_mask,
            "parking_mask": parking_mask,
            "hard_surface": hard_surface,
        },
    )


def _aggregate_points(
    frame: pd.DataFrame,
    area: StudyArea,
    size: int,
    value_column: str | None,
    agg: str,
    capabilities: RuntimeCapabilities,
) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        zeros = np.zeros((size, size), dtype=float)
        return zeros, zeros

    working = frame.copy()
    working["col"] = np.clip(
        ((working["longitude"] - area.lon_min) / (area.lon_max - area.lon_min) * size).astype(int),
        0,
        size - 1,
    )
    working["row"] = np.clip(
        ((area.lat_max - working["latitude"]) / (area.lat_max - area.lat_min) * size).astype(int),
        0,
        size - 1,
    )

    if capabilities.cudf:
        try:
            import cudf  # type: ignore

            gpu_frame = cudf.from_pandas(working[["row", "col"] + ([value_column] if value_column else [])])
            if value_column and agg == "mean":
                grouped = gpu_frame.groupby(["row", "col"])[value_column].mean().reset_index().to_pandas()
            else:
                grouped = gpu_frame.groupby(["row", "col"]).size().reset_index(name="value").to_pandas()
        except Exception:
            grouped = _group_cpu(working, value_column, agg)
    else:
        grouped = _group_cpu(working, value_column, agg)

    grid = np.zeros((size, size), dtype=float)
    density = np.zeros((size, size), dtype=float)

    for record in grouped.to_dict(orient="records"):
        row = int(record["row"])
        col = int(record["col"])
        value = float(record.get(value_column if value_column and agg == "mean" else "value", 0.0))
        grid[row, col] = value
        density[row, col] += 1.0

    return grid, density


def _group_cpu(frame: pd.DataFrame, value_column: str | None, agg: str) -> pd.DataFrame:
    if value_column and agg == "mean":
        return frame.groupby(["row", "col"], as_index=False)[value_column].mean()
    grouped = frame.groupby(["row", "col"], as_index=False).size()
    return grouped.rename(columns={"size": "value"})


def _normalize(array: np.ndarray, invert: bool = False) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=float)
    low = float(np.min(arr[valid]))
    high = float(np.max(arr[valid]))
    if np.isclose(high, low):
        normalized = np.zeros_like(arr, dtype=float)
    else:
        normalized = (arr - low) / (high - low)
    return 1.0 - normalized if invert else normalized


def _block_label_from_grid(
    primary_grid: np.ndarray,
    secondary_grid: np.ndarray,
    row_start: int,
    col_start: int,
    block_span: int,
    block_id: int,
) -> str:
    primary = primary_grid[row_start:row_start + block_span, col_start:col_start + block_span]
    secondary = secondary_grid[row_start:row_start + block_span, col_start:col_start + block_span]
    names: list[str] = []
    for grid in (primary, secondary):
        values, counts = np.unique(grid[grid != ""], return_counts=True)
        ordered = sorted(zip(counts.tolist(), values.tolist()), reverse=True)
        for _, value in ordered:
            if value not in names:
                names.append(value)
            if len(names) == 2:
                break
        if len(names) == 2:
            break

    if len(names) >= 2:
        return f"{names[0]} & {names[1]}"
    if len(names) == 1:
        return f"{names[0]} area"
    return f"Lower Manhattan Block {block_id + 1}"


def rasterize_centerlines(frame: pd.DataFrame, area: StudyArea, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    grid = np.zeros((size, size), dtype=float)
    primary_names = np.full((size, size), "", dtype=object)
    secondary_names = np.full((size, size), "", dtype=object)
    segments: list[dict[str, Any]] = []
    if frame.empty:
        return grid, primary_names, secondary_names, segments

    lon_span = area.lon_max - area.lon_min
    lat_span = area.lat_max - area.lat_min
    for record in frame.to_dict(orient="records"):
        lines = (record.get("coordinates") or []) if record.get("geometry_type") == "MultiLineString" else []
        width_m = max(float(record.get("streetwidth") or 0.0), 20.0)
        radius_cells = max(1, int(round(width_m / 20.0 / 2.0)))
        for line in lines:
            for start, end in zip(line, line[1:]):
                latlon_coordinates = [[start[1], start[0]], [end[1], end[0]]]
                x0 = int(np.clip(((start[0] - area.lon_min) / lon_span) * size, 0, size - 1))
                y0 = int(np.clip(((area.lat_max - start[1]) / lat_span) * size, 0, size - 1))
                x1 = int(np.clip(((end[0] - area.lon_min) / lon_span) * size, 0, size - 1))
                y1 = int(np.clip(((area.lat_max - end[1]) / lat_span) * size, 0, size - 1))
                touched_cells: set[tuple[int, int]] = set()
                for row, col in _bresenham(y0, x0, y1, x1):
                    r0 = max(0, row - radius_cells)
                    r1 = min(size, row + radius_cells + 1)
                    c0 = max(0, col - radius_cells)
                    c1 = min(size, col + radius_cells + 1)
                    grid[r0:r1, c0:c1] = 1.0
                    _assign_street_name(primary_names, secondary_names, r0, r1, c0, c1, record.get("street_name") or "")
                    touched_cells.add((row, col))
                if touched_cells:
                    segments.append(
                        {
                            "name": record.get("street_name") or "Unknown",
                            "coordinates": latlon_coordinates,
                            "cells": sorted(touched_cells),
                        }
                    )
    return grid, primary_names, secondary_names, segments


def _assign_street_name(
    primary_names: np.ndarray,
    secondary_names: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    street_name: str,
) -> None:
    name = street_name.strip()
    if not name:
        return
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            if primary_names[row, col] == "":
                primary_names[row, col] = name
            elif primary_names[row, col] != name and secondary_names[row, col] == "":
                secondary_names[row, col] = name


def _segment_label_grid(primary_names: np.ndarray, secondary_names: np.ndarray) -> np.ndarray:
    labels = np.full(primary_names.shape, "", dtype=object)
    rows, cols = primary_names.shape
    for row in range(rows):
        for col in range(cols):
            primary = str(primary_names[row, col]).strip()
            secondary = str(secondary_names[row, col]).strip()
            if primary and secondary:
                labels[row, col] = f"{primary} & {secondary}"
            elif primary:
                labels[row, col] = f"{primary} area"
    return labels


def rasterize_polygons(frame: pd.DataFrame, area: StudyArea, size: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=float)
    if frame.empty:
        return grid

    centers = [
        (
            row,
            col,
            area.lon_min + ((col + 0.5) / size) * (area.lon_max - area.lon_min),
            area.lat_max - ((row + 0.5) / size) * (area.lat_max - area.lat_min),
        )
        for row in range(size)
        for col in range(size)
    ]

    for record in frame.to_dict(orient="records"):
        polygons = (record.get("coordinates") or []) if record.get("geometry_type") == "MultiPolygon" else []
        for polygon in polygons:
            if not polygon:
                continue
            ring = polygon[0]
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            for row, col, lon, lat in centers:
                if lon < min_x or lon > max_x or lat < min_y or lat > max_y:
                    continue
                if _point_in_polygon(lon, lat, ring):
                    grid[row, col] = 1.0
    return grid


def _bresenham(y0: int, x0: int, y1: int, x1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield y0, x0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _point_in_polygon(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside
