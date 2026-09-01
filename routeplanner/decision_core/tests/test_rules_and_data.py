from __future__ import annotations

import unittest

from blackflow_rl.client_data import DEFAULT_CLIENT_DATA, validate_client_data
from blackflow_rl.domain import ResourceState
from blackflow_rl.rules import load_ruleset


class RulesAndDataTests(unittest.TestCase):
    def test_ruleset_has_verified_action_points(self) -> None:
        rules = load_ruleset()
        self.assertEqual(
            [rules.floor(floor).action_points for floor in range(1, 7)],
            [5, 6, 7, 8, 8, 5],
        )
        self.assertEqual(rules.action_size, 56)
        self.assertEqual(rules.schema_version, 1)
        self.assertEqual(rules.compatibility_profile, "lubiao-2026-08-31")
        self.assertEqual(
            rules.random_generation_status,
            "synthetic-only-explicit-opt-in",
        )
        self.assertEqual(
            rules.node_distance_metric,
            "lubiao-final-neighbour-shortcut-v1",
        )
        self.assertTrue(rules.map_templates_path and rules.map_templates_path.is_file())
        self.assertEqual(rules.objective.value_scale, 256.0)
        initial = ResourceState()
        self.assertEqual(
            (initial.hp, initial.max_hp, initial.gold, initial.hope, initial.parts),
            (8, 8, 8, 6, 0),
        )

    @unittest.skipUnless(
        DEFAULT_CLIENT_DATA.is_file(),
        "requires the optional complete client-data table",
    )
    def test_complete_client_table_golden_summary(self) -> None:
        summary = validate_client_data()
        self.assertEqual(
            summary.sha256,
            "aa2b1fc6ba0cc9ee29b9e6a08803550181c3a27189ac449efbad87608880d35b",
        )
        self.assertEqual(
            summary.git_blob_sha,
            "723f15432e989b6d0d402c38548a74a317f2f97c",
        )
        self.assertEqual(summary.node_types, 21)
        self.assertEqual(summary.choices, 396)
        self.assertEqual(summary.choice_scenes, 338)
        self.assertEqual(summary.stages, 105)
        self.assertEqual(
            (
                summary.normal_stages,
                summary.elite_stages,
                summary.boss_stages,
                summary.special_stages,
                summary.chase_stages,
                summary.duel_stages,
            ),
            (31, 34, 10, 19, 9, 2),
        )
        self.assertEqual((summary.move_scraps, summary.goods_scraps, summary.passive_scraps), (12, 12, 6))

    @unittest.skipUnless(
        (DEFAULT_CLIENT_DATA.parent / "roguelike_topic_table.json").is_file(),
        "requires the optional transport-fragment fixture",
    )
    def test_loader_rejects_transport_fragment(self) -> None:
        fragment = DEFAULT_CLIENT_DATA.parent / "roguelike_topic_table.json"
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_client_data(fragment)


if __name__ == "__main__":
    unittest.main()
