from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from blackflow.config import load_config
from blackflow.simulator import run_simulation


ROOT = Path(__file__).resolve().parents[1]


class SimulatorTests(unittest.TestCase):
    def test_closed_loop_prefers_safe_plan(self) -> None:
        config = load_config(ROOT / "configs" / "strategy_first_battle.example.json")
        with tempfile.TemporaryDirectory() as temp:
            config.data["simulation"]["state_file"] = str(Path(temp) / "sim_q.json")
            rows = run_simulation(config, episodes=300, seed=11)
            values = {row["plan"]: row for row in rows}
            safe = values["safe_block_heal_damage"]
            early = values["early_damage_then_block"]
            self.assertGreater(safe["q"], early["q"])
            self.assertGreater(safe["visits"], early["visits"])


if __name__ == "__main__":
    unittest.main()
