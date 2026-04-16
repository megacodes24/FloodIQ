from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from .settings import CACHE_DIR, StudyArea


SOCRATA_BASE = "https://data.cityofnewyork.us/resource"
SEWER_DATASET_ID = "erm2-nwe9"
ELEVATION_DATASET_ID = "9uxf-ng6q"
STREET_CENTERLINE_DATASET_ID = "inkn-q76z"
SIDEWALK_DATASET_ID = "52n9-sdep"
PARKING_LOT_DATASET_ID = "7cgt-uhhz"
CATCH_BASIN_DATASET_ID = "2w2g-fk3i"
OUTFALL_DATASET_ID = "8rjn-kpsh"


@dataclass(frozen=True)
class NYCDataBundle:
    sewer_complaints: pd.DataFrame
    elevation_points: pd.DataFrame
    street_centerlines: pd.DataFrame
    sidewalk_polygons: pd.DataFrame
    parking_lot_polygons: pd.DataFrame
    catch_basins: pd.DataFrame
    outfalls: pd.DataFrame
    metadata: dict[str, Any]


class NYCOpenDataClient:
    def __init__(self, cache_dir: Path | None = None, timeout_seconds: int = 30) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.timeout_seconds = timeout_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bundle(
        self,
        area: StudyArea,
        complaint_limit: int = 25000,
        elevation_limit: int = 30000,
        refresh: bool = False,
    ) -> NYCDataBundle:
        complaints = self.fetch_sewer_complaints(area, complaint_limit, refresh=refresh)
        elevation = self.fetch_elevation_points(area, elevation_limit, refresh=refresh)
        centerlines = self.fetch_street_centerlines(area, refresh=refresh)
        sidewalks = self.fetch_polygon_layer(SIDEWALK_DATASET_ID, area, "sidewalk", refresh=refresh)
        parking_lots = self.fetch_polygon_layer(PARKING_LOT_DATASET_ID, area, "parking_lots", refresh=refresh)
        catch_basins = self.fetch_point_layer(CATCH_BASIN_DATASET_ID, area, "catch_basins", refresh=refresh)
        outfalls = self.fetch_point_layer(OUTFALL_DATASET_ID, area, "outfalls", refresh=refresh)
        return NYCDataBundle(
            sewer_complaints=complaints,
            elevation_points=elevation,
            street_centerlines=centerlines,
            sidewalk_polygons=sidewalks,
            parking_lot_polygons=parking_lots,
            catch_basins=catch_basins,
            outfalls=outfalls,
            metadata={
                "complaint_rows": int(len(complaints)),
                "elevation_rows": int(len(elevation)),
                "street_centerline_rows": int(len(centerlines)),
                "sidewalk_polygon_rows": int(len(sidewalks)),
                "parking_lot_polygon_rows": int(len(parking_lots)),
                "catch_basin_rows": int(len(catch_basins)),
                "outfall_rows": int(len(outfalls)),
                "area": area.name,
                "sources": {
                    "sewer_complaints_dataset": SEWER_DATASET_ID,
                    "elevation_dataset": ELEVATION_DATASET_ID,
                    "street_centerline_dataset": STREET_CENTERLINE_DATASET_ID,
                    "sidewalk_dataset": SIDEWALK_DATASET_ID,
                    "parking_lot_dataset": PARKING_LOT_DATASET_ID,
                    "catch_basin_dataset": CATCH_BASIN_DATASET_ID,
                    "outfall_dataset": OUTFALL_DATASET_ID,
                },
            },
        )

    def fetch_sewer_complaints(
        self,
        area: StudyArea,
        limit: int = 25000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"sewer_complaints_{area.slug}_{limit}.json"
        params = {
            "$select": ",".join(
                [
                    "created_date",
                    "complaint_type",
                    "descriptor",
                    "borough",
                    "community_board",
                    "latitude",
                    "longitude",
                ]
            ),
            "$where": (
                "complaint_type='Sewer' "
                f"AND latitude IS NOT NULL AND longitude IS NOT NULL "
                f"AND longitude > {area.lon_min} AND longitude < {area.lon_max} "
                f"AND latitude > {area.lat_min} AND latitude < {area.lat_max}"
            ),
            "$order": "created_date DESC",
            "$limit": str(limit),
        }
        records = self._get_json(SEWER_DATASET_ID, params, cache_path, refresh=refresh)
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return pd.DataFrame(
                columns=["created_date", "complaint_type", "descriptor", "borough", "community_board", "latitude", "longitude"]
            )
        frame["created_date"] = pd.to_datetime(frame["created_date"], errors="coerce", utc=True)
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame = frame.dropna(subset=["created_date", "latitude", "longitude"]).reset_index(drop=True)
        return frame

    def fetch_elevation_points(
        self,
        area: StudyArea,
        limit: int = 30000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"elevation_points_{area.slug}_{limit}.json"
        params = {
            "$select": "elevation,the_geom",
            "$where": (
                f"within_box(the_geom, {area.lat_min}, {area.lon_min}, {area.lat_max}, {area.lon_max})"
            ),
            "$limit": str(limit),
        }
        records = self._get_json(ELEVATION_DATASET_ID, params, cache_path, refresh=refresh)
        rows: list[dict[str, float]] = []
        for record in records:
            geom = record.get("the_geom") or {}
            coords = geom.get("coordinates") or []
            if len(coords) != 2:
                continue
            try:
                rows.append(
                    {
                        "elevation": float(record["elevation"]),
                        "longitude": float(coords[0]),
                        "latitude": float(coords[1]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return pd.DataFrame(rows, columns=["elevation", "longitude", "latitude"])

    def fetch_street_centerlines(
        self,
        area: StudyArea,
        limit: int = 12000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"street_centerlines_{area.slug}_{limit}.json"
        params = {
            "$select": "the_geom,streetwidth,street_name,full_street_name,number_total_lanes",
            "$where": f"within_box(the_geom, {area.lat_min}, {area.lon_min}, {area.lat_max}, {area.lon_max})",
            "$limit": str(limit),
        }
        records = self._get_json(STREET_CENTERLINE_DATASET_ID, params, cache_path, refresh=refresh)
        rows: list[dict[str, Any]] = []
        for record in records:
            geom = record.get("the_geom") or {}
            coords = geom.get("coordinates") or []
            if not coords:
                continue
            rows.append(
                {
                    "geometry_type": geom.get("type"),
                    "coordinates": coords,
                    "streetwidth": _to_float(record.get("streetwidth")) or 0.0,
                    "street_name": record.get("street_name") or record.get("full_street_name") or "Unknown",
                    "number_total_lanes": _to_float(record.get("number_total_lanes")) or 0.0,
                }
            )
        return pd.DataFrame(rows)

    def fetch_polygon_layer(
        self,
        dataset_id: str,
        area: StudyArea,
        cache_prefix: str,
        limit: int = 12000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{cache_prefix}_{area.slug}_{limit}.json"
        params = {
            "$select": "the_geom,shape_area",
            "$where": f"within_box(the_geom, {area.lat_min}, {area.lon_min}, {area.lat_max}, {area.lon_max})",
            "$limit": str(limit),
        }
        records = self._get_json(dataset_id, params, cache_path, refresh=refresh)
        rows: list[dict[str, Any]] = []
        for record in records:
            geom = record.get("the_geom") or {}
            coords = geom.get("coordinates") or []
            if not coords:
                continue
            rows.append(
                {
                    "geometry_type": geom.get("type"),
                    "coordinates": coords,
                    "shape_area": _to_float(record.get("shape_area")) or 0.0,
                }
            )
        return pd.DataFrame(rows)

    def fetch_point_layer(
        self,
        dataset_id: str,
        area: StudyArea,
        cache_prefix: str,
        limit: int = 12000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{cache_prefix}_{area.slug}_{limit}.json"
        params = {
            "$select": "the_geom,latitude,longitude,unitid",
            "$where": (
                f"longitude > {area.lon_min} AND longitude < {area.lon_max} "
                f"AND latitude > {area.lat_min} AND latitude < {area.lat_max}"
            ),
            "$limit": str(limit),
        }
        records = self._get_json(dataset_id, params, cache_path, refresh=refresh)
        rows: list[dict[str, Any]] = []
        for record in records:
            geom = record.get("the_geom") or {}
            coords = geom.get("coordinates") or []
            lon = _to_float(record.get("longitude"))
            lat = _to_float(record.get("latitude"))
            if lon is None or lat is None:
                if len(coords) == 2:
                    lon = _to_float(coords[0])
                    lat = _to_float(coords[1])
            if lon is None or lat is None:
                continue
            rows.append(
                {
                    "unitid": record.get("unitid") or "unknown",
                    "longitude": lon,
                    "latitude": lat,
                }
            )
        return pd.DataFrame(rows, columns=["unitid", "longitude", "latitude"])

    def _get_json(
        self,
        dataset_id: str,
        params: dict[str, str],
        cache_path: Path,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text())

        url = f"{SOCRATA_BASE}/{dataset_id}.json?{urlencode(params)}"
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        records = response.json()
        cache_path.write_text(json.dumps(records))
        return records


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
