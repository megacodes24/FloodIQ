from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .noaa import NOAAClient
from .service import build_service_for_slug
from .settings import STUDY_AREAS


STATIC_DIR = Path(__file__).with_name("static")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    services: dict[str, object] = {}

    def get_service(study_area_slug: str | None, refresh: bool = False):
        key = study_area_slug or "nyc"
        if refresh or key not in services:
            services[key] = build_service_for_slug(study_area_slug)
        return services[key]

    def forecast_board(refresh: bool = False) -> dict[str, object]:
        preferred = ["lower_manhattan", "manhattan", "gowanus", "east_elmhurst", "southeast_queens"]
        noaa = NOAAClient()
        areas: list[dict[str, object]] = []
        for slug in preferred:
            area = STUDY_AREAS[slug]
            forecast = None
            status = "No rain forecast currently available"
            try:
                raw_forecast = noaa.fetch_quantitative_precipitation(*area.center)
                if raw_forecast is not None:
                    forecast = {
                        "name": raw_forecast.name,
                        "rainfall_inches_per_hour": raw_forecast.rainfall_inches_per_hour,
                        "duration_hours": raw_forecast.duration_hours,
                        "valid_time": raw_forecast.valid_time,
                        "start_time_iso": raw_forecast.start_time_iso,
                        "start_time_label": raw_forecast.start_time_label,
                        "hours_until_start": raw_forecast.hours_until_start,
                        "source_summary": raw_forecast.source_summary,
                        "precipitation_probability": raw_forecast.precipitation_probability,
                    }
                    status = "NOAA forecast ready"
            except Exception as exc:
                status = f"NOAA unavailable: {exc}"
            areas.append(
                {
                    "slug": slug,
                    "name": area.name,
                    "forecast_available": forecast is not None,
                    "forecast": forecast,
                    "status": status,
                    "subtitle": "Flood-prone pilot zone" if slug != "manhattan" else "Wider Manhattan screening view",
                }
            )
        return {"areas": areas}

    class FloodIQHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            study_area_slug = query.get("study_area", [None])[0]
            refresh = query.get("refresh", ["0"])[0] == "1"
            if parsed.path == "/api/baseline":
                service = get_service(study_area_slug, refresh=refresh)
                self._send_json(service.baseline_payload())
                return
            if parsed.path == "/api/forecast_board":
                self._send_json(forecast_board(refresh=refresh))
                return
            if parsed.path == "/":
                self._send_file("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._send_file("app.js", "application/javascript; charset=utf-8")
                return
            if parsed.path == "/styles.css":
                self._send_file("styles.css", "text/css; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/simulate":
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_payload = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw_payload)

            rainfall_raw = payload.get("rainfall_inches_per_hour")
            duration_raw = payload.get("duration_hours")
            rainfall = float(rainfall_raw) if rainfall_raw is not None else None
            duration = float(duration_raw) if duration_raw is not None else None
            name = str(payload.get("name", "Custom scenario"))
            study_area_slug = payload.get("study_area")
            refresh = bool(payload.get("refresh"))
            service = get_service(study_area_slug, refresh=refresh)
            response = service.run_scenario(rainfall, duration, name)
            self._send_json(response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, filename: str, content_type: str) -> None:
            path = STATIC_DIR / filename
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), FloodIQHandler)
    print(f"FloodIQ listening on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FloodIQ local web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
