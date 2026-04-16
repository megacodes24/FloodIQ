from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests


POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"


@dataclass(frozen=True)
class ForecastScenario:
    name: str
    rainfall_inches_per_hour: float
    duration_hours: float
    source_url: str
    valid_time: str
    start_time_iso: str
    start_time_label: str
    hours_until_start: float
    source_summary: str
    precipitation_probability: int


class NOAAClient:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": "FloodIQ/0.1 (hackathon prototype contact: local-demo)",
            "Accept": "application/geo+json,application/json",
        }

    def fetch_quantitative_precipitation(self, lat: float, lon: float) -> ForecastScenario | None:
        point_response = requests.get(
            POINTS_URL.format(lat=lat, lon=lon),
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        point_response.raise_for_status()
        point_payload = point_response.json()
        grid_url = point_payload["properties"]["forecastGridData"]
        hourly_url = point_payload["properties"].get("forecastHourly")

        grid_response = requests.get(grid_url, headers=self.headers, timeout=self.timeout_seconds)
        grid_response.raise_for_status()
        grid_payload = grid_response.json()

        hourly_payload: dict[str, Any] | None = None
        if hourly_url:
            hourly_response = requests.get(hourly_url, headers=self.headers, timeout=self.timeout_seconds)
            hourly_response.raise_for_status()
            hourly_payload = hourly_response.json()

        qpf_values = grid_payload["properties"].get("quantitativePrecipitation", {}).get("values", [])
        qpf_periods = [self._parse_qpf_entry(entry, grid_url) for entry in qpf_values]
        qpf_periods = [period for period in qpf_periods if period is not None]

        hourly_periods = (hourly_payload or {}).get("properties", {}).get("periods", [])
        for period in hourly_periods:
            candidate = self._parse_hourly_period(period, qpf_periods, hourly_url or grid_url)
            if candidate is not None:
                return candidate

        for period in qpf_periods:
            return period
        return None

    def _parse_qpf_entry(self, entry: dict[str, Any], source_url: str) -> ForecastScenario | None:
        value = entry.get("value")
        if value in (None, 0):
            return None
        valid_time = entry.get("validTime", "")
        duration_hours = self._duration_hours(valid_time)
        if duration_hours <= 0:
            return None
        start_time = self._start_time(valid_time)
        if start_time is None:
            return None
        mm_to_inches = 0.0393701
        inches_per_hour = (float(value) * mm_to_inches) / duration_hours
        return self._build_forecast_scenario(
            rainfall_inches_per_hour=inches_per_hour,
            duration_hours=duration_hours,
            source_url=source_url,
            valid_time=valid_time,
            source_summary="NOAA quantitative precipitation forecast",
            precipitation_probability=100,
        )

    def _parse_hourly_period(
        self,
        period: dict[str, Any],
        qpf_periods: list[ForecastScenario],
        source_url: str,
    ) -> ForecastScenario | None:
        start_text = str(period.get("startTime") or "")
        end_text = str(period.get("endTime") or "")
        start_time = self._start_time(start_text)
        end_time = self._start_time(end_text)
        if start_time is None or end_time is None:
            return None
        probability = int((period.get("probabilityOfPrecipitation") or {}).get("value") or 0)
        summary = str(period.get("shortForecast") or "")
        if probability < 25 and not self._looks_rainy(summary):
            return None

        matching_qpf = next((item for item in qpf_periods if item.start_time_iso == start_time.isoformat()), None)
        duration_hours = max(1.0 / 6.0, (end_time - start_time).total_seconds() / 3600.0)
        if matching_qpf is not None:
            return ForecastScenario(
                name="Live NOAA forecast",
                rainfall_inches_per_hour=matching_qpf.rainfall_inches_per_hour,
                duration_hours=matching_qpf.duration_hours,
                source_url=source_url,
                valid_time=f"{start_text}/PT{max(1, round(duration_hours))}H",
                start_time_iso=matching_qpf.start_time_iso,
                start_time_label=matching_qpf.start_time_label,
                hours_until_start=matching_qpf.hours_until_start,
                source_summary=summary or matching_qpf.source_summary,
                precipitation_probability=probability,
            )

        estimated_mm_per_hour = self._estimate_rainfall_mm_per_hour(summary, probability)
        return self._build_forecast_scenario(
            rainfall_inches_per_hour=estimated_mm_per_hour * 0.0393701,
            duration_hours=duration_hours,
            source_url=source_url,
            valid_time=f"{start_text}/PT{max(1, round(duration_hours))}H",
            source_summary=summary or "NOAA hourly rain forecast",
            precipitation_probability=probability,
        )

    def _build_forecast_scenario(
        self,
        rainfall_inches_per_hour: float,
        duration_hours: float,
        source_url: str,
        valid_time: str,
        source_summary: str,
        precipitation_probability: int,
    ) -> ForecastScenario:
        start_time = self._start_time(valid_time)
        assert start_time is not None
        local_start = start_time.astimezone(ZoneInfo("America/New_York"))
        hours_until_start = max(0.0, (start_time - datetime.now(start_time.tzinfo)).total_seconds() / 3600.0)
        return ForecastScenario(
            name="Live NOAA forecast",
            rainfall_inches_per_hour=round(rainfall_inches_per_hour, 2),
            duration_hours=round(duration_hours, 2),
            source_url=source_url,
            valid_time=valid_time,
            start_time_iso=start_time.isoformat(),
            start_time_label=local_start.strftime("%b %d, %Y at %-I:%M %p %Z"),
            hours_until_start=round(hours_until_start, 1),
            source_summary=source_summary,
            precipitation_probability=precipitation_probability,
        )

    @staticmethod
    def _looks_rainy(summary: str) -> bool:
        lowered = summary.lower()
        return any(keyword in lowered for keyword in ("rain", "shower", "storm", "thunder"))

    @staticmethod
    def _estimate_rainfall_mm_per_hour(summary: str, probability: int) -> float:
        lowered = summary.lower()
        if "heavy" in lowered or "thunder" in lowered:
            return 9.0
        if "rain" in lowered or "showers" in lowered:
            return 5.0 if probability >= 60 else 2.5
        if probability >= 70:
            return 3.0
        return 1.5

    @staticmethod
    def _duration_hours(valid_time: str) -> float:
        match = re.search(r"/P(?:([0-9]+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$", valid_time)
        if not match:
            return 0.0
        days = int(match.group(1) or 0)
        hours = int(match.group(2) or 0)
        minutes = int(match.group(3) or 0)
        return (days * 24) + hours + (minutes / 60.0)

    @staticmethod
    def _start_time(valid_time: str) -> datetime | None:
        start_text = valid_time.split("/", 1)[0].strip()
        if not start_text:
            return None
        try:
            return datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        except ValueError:
            return None
