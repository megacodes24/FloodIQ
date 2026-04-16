from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

NYC_BBOX = {
    "lon_min": -74.25515956631524,
    "lon_max": -73.69964757062174,
    "lat_min": 40.49656773180599,
    "lat_max": 40.915101939643485,
}


@dataclass(frozen=True)
class StudyArea:
    name: str
    slug: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.lat_min + self.lat_max) / 2.0, (self.lon_min + self.lon_max) / 2.0)


DEFAULT_STUDY_AREA = StudyArea(
    name="New York City",
    slug="nyc",
    lon_min=NYC_BBOX["lon_min"],
    lon_max=NYC_BBOX["lon_max"],
    lat_min=NYC_BBOX["lat_min"],
    lat_max=NYC_BBOX["lat_max"],
)

STUDY_AREAS = {
    "nyc": DEFAULT_STUDY_AREA,
    "gowanus": StudyArea(
        name="Gowanus",
        slug="gowanus",
        lon_min=-74.013,
        lon_max=-73.981,
        lat_min=40.655,
        lat_max=40.690,
    ),
    "east_elmhurst": StudyArea(
        name="East Elmhurst",
        slug="east_elmhurst",
        lon_min=-73.902,
        lon_max=-73.855,
        lat_min=40.752,
        lat_max=40.780,
    ),
    "southeast_queens": StudyArea(
        name="Southeast Queens",
        slug="southeast_queens",
        lon_min=-73.808,
        lon_max=-73.735,
        lat_min=40.665,
        lat_max=40.718,
    ),
    "lower_manhattan": StudyArea(
        name="Lower Manhattan",
        slug="lower_manhattan",
        lon_min=-74.025,
        lon_max=-73.968,
        lat_min=40.700,
        lat_max=40.735,
    ),
    "manhattan": StudyArea(
        name="All Manhattan",
        slug="manhattan",
        lon_min=-74.0479,
        lon_max=-73.9067,
        lat_min=40.6829,
        lat_max=40.8820,
    ),
}


def get_study_area(slug: str | None) -> StudyArea:
    if slug is None:
        return DEFAULT_STUDY_AREA
    return STUDY_AREAS.get(slug, DEFAULT_STUDY_AREA)
