from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from blackflow.learner import QLearner


class LearnerTests(unittest.TestCase):
    def test_learns_better_plan_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "q.json"
            learner = QLearner(path, alpha=0.2, epsilon=0.25, epsilon_decay=0.99, seed=3)
            rng = random.Random(3)
            rates = {"bad": 0.1, "good": 0.9}
            for _ in range(500):
                plan = learner.select("ctx", rates, training=True)
                won = rng.random() < rates[plan]
                learner.update("ctx", plan, 1.0 if won else -1.0, won=won)
            rows = {row["plan"]: row for row in learner.summary("ctx")}
            self.assertGreater(rows["good"]["q"], rows["bad"]["q"])
            self.assertGreater(rows["good"]["visits"], rows["bad"]["visits"])
            reloaded = QLearner(path)
            self.assertEqual(reloaded.episodes, 500)
            self.assertEqual(reloaded.summary("ctx"), learner.summary("ctx"))


if __name__ == "__main__":
    unittest.main()

