from __future__ import annotations

import unittest
from pathlib import Path

from blackflow.config import load_config
from blackflow.vision import Detection, evaluate_condition, resolve_point


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndVisionTests(unittest.TestCase):
    def test_example_config_is_valid(self) -> None:
        config = load_config(ROOT / "configs" / "strategy_first_battle.example.json")
        self.assertEqual(config.data["base_resolution"], [1920, 1080])
        self.assertEqual(len(config.data["plans"]), 3)

    def test_conditions_and_detector_points(self) -> None:
        detections = {
            "ready": Detection("ready", True, 0.95, (100.0, 200.0)),
            "danger": Detection("danger", False, 0.10, None),
        }
        condition = {
            "all": [
                {"elapsed_ge": 3.0},
                {"detector": "ready"},
                {"not": {"detector": "danger"}},
            ]
        }
        self.assertTrue(evaluate_condition(condition, detections, 4.0))
        self.assertFalse(evaluate_condition(condition, detections, 2.0))
        self.assertEqual(resolve_point({"detector": "ready", "offset": [5, -5]}, {}, detections), [105.0, 195.0])


if __name__ == "__main__":
    unittest.main()

