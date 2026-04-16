from __future__ import annotations

from pprint import pprint

from .service import build_service_for_slug


def main() -> None:
    service = build_service_for_slug("lower_manhattan", use_live_data=True)
    baseline = service.baseline_payload()
    scenario = service.run_scenario(3.15, 1.0, "Hurricane Ida replay")

    print("=== Baseline ===")
    pprint(
        {
            "data_mode": baseline["data_mode"],
            "solver": baseline["solver"],
            "validation": baseline["evaluation"]["validation"],
            "benchmark": baseline["evaluation"]["benchmark"],
        }
    )
    print("\n=== Scenario ===")
    pprint(
        {
            "summary": scenario["summary"],
            "recommended_actions": scenario["recommended_actions"],
        }
    )


if __name__ == "__main__":
    main()
