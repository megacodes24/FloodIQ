import unittest

from floodiq.solver_backends.factory import build_solver_backend
from floodiq.service import FloodIQService, build_service_for_slug
from floodiq.settings import DEFAULT_STUDY_AREA, get_study_area
from floodiq.real_grid import _normalize


class FloodIQServiceTests(unittest.TestCase):
    def test_more_rain_creates_more_flooding(self) -> None:
        service = FloodIQService(use_live_data=False)
        light = service.run_scenario(0.8, 1.0, "Light")
        heavy = service.run_scenario(3.2, 1.0, "Heavy")

        self.assertGreater(
            heavy["summary"]["peak_depth_m"],
            light["summary"]["peak_depth_m"],
        )
        self.assertGreater(
            heavy["summary"]["flooded_cells"],
            light["summary"]["flooded_cells"],
        )

    def test_blocks_are_ranked(self) -> None:
        service = FloodIQService(use_live_data=False)
        result = service.run_scenario(2.0, 1.5, "Ranked")
        top = result["blocks"][0]
        bottom = result["blocks"][-1]

        self.assertGreaterEqual(top["risk_score"], bottom["risk_score"])
        self.assertIn("top_block", result["summary"])

    def test_normalize_handles_flat_arrays(self) -> None:
        result = _normalize([[3.0, 3.0], [3.0, 3.0]])
        self.assertEqual(float(result.max()), 0.0)

    def test_service_has_a_study_area(self) -> None:
        self.assertEqual(DEFAULT_STUDY_AREA.name, "New York City")
        self.assertEqual(get_study_area("gowanus").name, "Gowanus")

    def test_solver_backend_is_available(self) -> None:
        solver = build_solver_backend()
        self.assertIn(solver.metadata.name, {"fallback", "physicsnemo"})
        self.assertIsInstance(solver.metadata.ready, bool)

    def test_service_builder_uses_requested_study_area(self) -> None:
        service = build_service_for_slug("lower_manhattan", use_live_data=False)
        self.assertEqual(service.study_area.slug, "lower_manhattan")

    def test_offline_service_still_runs_with_new_real_grid_signature(self) -> None:
        service = build_service_for_slug("lower_manhattan", use_live_data=False)
        result = service.run_scenario(3.15, 1.0)
        self.assertIn("summary", result)
        self.assertIn("recommended_actions", result)
        self.assertTrue(result["recommended_actions"])

    def test_baseline_includes_evaluation_payload(self) -> None:
        service = build_service_for_slug("lower_manhattan", use_live_data=False)
        baseline = service.baseline_payload()
        self.assertIn("evaluation", baseline)
        self.assertIn("validation", baseline["evaluation"])
        self.assertIn("benchmark", baseline["evaluation"])


if __name__ == "__main__":
    unittest.main()
