from __future__ import annotations

from pprint import pprint

from .noaa import NOAAClient
from .nyc_open_data import NYCOpenDataClient
from .settings import DEFAULT_STUDY_AREA


def main() -> None:
    nyc = NYCOpenDataClient()
    bundle = nyc.fetch_bundle(DEFAULT_STUDY_AREA, refresh=True)
    forecast = NOAAClient().fetch_quantitative_precipitation(40.7128, -74.0060)

    pprint(bundle.metadata)
    pprint(forecast)


if __name__ == "__main__":
    main()
