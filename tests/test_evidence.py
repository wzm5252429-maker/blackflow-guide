from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

import blackflow_rl.evidence as evidence_module
from blackflow_rl.evidence import (
    CLIENT_SNAPSHOT,
    CLIENT_SOURCE,
    EVENT_CATALOG,
    MAP_CONFLICTS,
    audit_evidence,
)
from blackflow_rl.domain import NodeType
from blackflow_rl.mapgen import _CONTENT_TOTAL_RANGES
from blackflow_rl.rules import load_ruleset


class EvidenceTests(unittest.TestCase):
    def test_pinned_evidence_artifacts_are_complete_and_integral(self) -> None:
        audit = audit_evidence()
        self.assertTrue(audit.integrity_ok)
        self.assertFalse(audit.verified_training_ready)
        self.assertEqual(audit.client_source_present, CLIENT_SOURCE.is_file())
        self.assertEqual(audit.client_choices, 396)
        self.assertEqual(audit.client_choice_scenes, 338)
        self.assertEqual(audit.event_groups, 38)
        self.assertEqual(audit.map_rule_conflicts, 26)
        self.assertTrue(audit.blockers)
        self.assertTrue(
            any("102 event summary branches" in item for item in audit.blockers)
        )

    def test_compact_published_snapshot_is_auditable_without_full_source(self) -> None:
        missing = Path(__file__).parent / "golden" / "not-present-client-source.json"
        with patch.object(evidence_module, "CLIENT_SOURCE", missing):
            audit = evidence_module.audit_evidence()
        self.assertTrue(audit.integrity_ok)
        self.assertFalse(audit.client_source_present)
        self.assertTrue(
            any("cannot be re-derived locally" in item for item in audit.blockers)
        )

    def test_client_golden_values_are_directly_pinned(self) -> None:
        snapshot = json.loads(CLIENT_SNAPSHOT.read_text(encoding="utf-8"))
        initial = snapshot["initial_normal_grade_zero"]
        self.assertEqual(
            (
                initial["initialHp"],
                initial["initialMaxHp"],
                initial["initialPopulation"],
                initial["initialGold"],
                initial["initialSquadCapacity"],
                initial["initialShield"],
            ),
            (8, 8, 6, 8, 6, 0),
        )
        choices = {item["id"]: item for item in snapshot["choices"]}
        scout_rest = choices["choice_ro6_scout_2"]
        self.assertEqual(scout_rest["title"], "休息")
        self.assertIn("2</>希望", scout_rest["description"])
        self.assertEqual(scout_rest["type"], "TRADE")
        self.assertEqual(scout_rest["nextSceneId"], "scene_ro6_scout_1")

    def test_dynamic_costs_and_exact_items_are_not_flattened(self) -> None:
        catalog = json.loads(EVENT_CATALOG.read_text(encoding="utf-8"))
        for node in catalog["node_rules"]:
            self.assertEqual(node["display_confidence"], "A")
            self.assertIn(node["mechanics_confidence"], {"B", "C"})
            self.assertEqual(node["confidence"], node["mechanics_confidence"])
        events = {item["name"]: item for item in catalog["events"]}
        heavy = events["沉重的契约"]
        self.assertEqual(heavy["options"][1]["cost"], "当前目标生命值减半")
        tears = events["泪之聚落"]
        self.assertEqual(tears["options"][0]["cost"], "失去全部源石锭")
        self.assertIn("击坠神明", tears["options"][0]["result"])
        capture = events["擒与缚"]
        self.assertEqual(
            [option["label"] for option in capture["options"]],
            ["选择『翱翼』", "选择『虬蜕』", "离开"],
        )
        self.assertNotIn("狭路相逢·右档", events)

    def test_active_map_profile_matches_lubiao_side_of_every_conflict(self) -> None:
        manifest = json.loads(MAP_CONFLICTS.read_text(encoding="utf-8"))
        self.assertEqual(manifest["active_compatibility_profile"], "lubiao-2026-08-31")
        rules = load_ruleset()
        for conflict in manifest["conflicts"]:
            node_rule = rules.node_rules[NodeType(conflict["node_type"])]
            floor = conflict["floor"]
            if conflict["field"] == "max_count":
                actual = node_rule.count_range(floor)[1]
            elif conflict["field"] == "min_distance":
                actual = node_rule.distance_range(floor)[0]
            else:
                actual = node_rule.distance_range(floor)[1]
            expected = conflict["lubiao"]
            if expected == "unbounded":
                expected = rules.floor(floor).maximum_graph_distance
            self.assertEqual(actual, expected, conflict)

    def test_runtime_count_and_distance_rules_are_parsed_from_raw_lubiao_fragments(self) -> None:
        fixture_path = Path(__file__).parent / "golden" / "lubiao_map_rules_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(
            sha256(fixture["count_fragment"].encode()).hexdigest(),
            fixture["count_fragment_sha256"],
        )
        self.assertEqual(
            sha256(fixture["distance_fragment"].encode()).hexdigest(),
            fixture["distance_fragment_sha256"],
        )
        self.assertEqual((len(fixture["counts"]), len(fixture["distances"])), (25, 25))

        source_to_runtime = {
            "rogue-trader": NodeType.BATTLE_SHOP,
            "secret-trader": NodeType.SCRAP_SHOP,
            "wrong-turn": NodeType.PORTAL,
            "narrow-meet": NodeType.DUEL,
            "林间空地": NodeType.EMPTY,
            "combat": NodeType.BATTLE_NORMAL,
            "emergency-combat": NodeType.BATTLE_ELITE,
            "danger-enemy": NodeType.BATTLE_BOSS,
            "settlement": NodeType.BATTLE_SAVAGE,
            "safe-corner": NodeType.REST,
            "encounter": NodeType.INCIDENT,
            "wish": NodeType.WISH,
            "gain-loss": NodeType.SACRIFICE,
            "first-step": NodeType.EXPEDITION,
            "命运所指（二结局）": NodeType.STORY,
            "emergency-aid": NodeType.EMPLOY,
            "winding-path": NodeType.DOOR,
            "overlook": NodeType.LIGHT,
            "danger-path": NodeType.EVACUATE,
            "danger-end": NodeType.FINAL,
        }
        rules = load_ruleset()

        def runtime_type(row: dict) -> NodeType | None:
            return source_to_runtime.get(row.get("node_type") or row["label"])

        for row in fixture["counts"]:
            node_type = runtime_type(row)
            if node_type is None:
                continue
            for floor, value in enumerate(row["values"], start=1):
                expected = (0, 0)
                if value is not None:
                    if value["kind"] == "set":
                        expected = (min(value["values"]), max(value["values"]))
                    else:
                        expected = (value["minimum"], value["maximum"])
                self.assertEqual(
                    rules.node_rules[node_type].count_range(floor),
                    expected,
                    (row["label"], floor),
                )

        distance_floor_indices = {1: 0, 2: 1, 3: 2, 4: 3, 5: 5}
        for row in fixture["distances"]:
            node_type = runtime_type(row)
            if node_type is None:
                continue
            for floor, index in distance_floor_indices.items():
                value = row["values"][index]
                if value is not None and value.get("kind") == "unknown":
                    continue
                expected = None
                if value is not None:
                    maximum = value["maximum"]
                    expected = (
                        value["minimum"],
                        rules.floor(floor).maximum_graph_distance
                        if maximum is None
                        else maximum,
                    )
                self.assertEqual(
                    rules.node_rules[node_type].distance_range(floor),
                    expected,
                    (row["label"], floor),
                )

        aggregates = {row["label"]: row for row in fixture["counts"]}
        for floor in range(1, 6):
            for category, label in (
                ("mystery", "诡秘类型节点"),
                ("ferocity", "凶戾类型节点"),
            ):
                value = aggregates[label]["values"][floor - 1]
                self.assertEqual(
                    _CONTENT_TOTAL_RANGES[floor][category],
                    (value["minimum"], value["maximum"]),
                )


if __name__ == "__main__":
    unittest.main()
