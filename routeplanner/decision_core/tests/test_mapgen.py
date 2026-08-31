from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
import unittest

from blackflow_rl.domain import NodeType
from blackflow_rl.mapgen import (
    MapGenerationError,
    MapGenerator,
    MapGeneratorConfig,
    _distances,
    _rule_distances,
)


class MapGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = MapGenerator()

    def test_seed_is_reproducible(self) -> None:
        first = self.generator.generate_run(1234)
        second = self.generator.generate_run(1234)
        self.assertEqual(
            [item.fingerprint for item in first],
            [item.fingerprint for item in second],
        )
        different = self.generator.generate_run(1235)
        self.assertNotEqual(first[0].fingerprint, different[0].fingerprint)

    def test_ending_route_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            MapGenerator(
                config=MapGeneratorConfig(
                    enable_second_ending=True,
                    enable_third_ending=True,
                )
            )

    def test_third_ending_explicitly_appends_fixed_sixth_floor(self) -> None:
        normal = self.generator.generate_run(20260831)
        third_generator = MapGenerator(
            config=MapGeneratorConfig(enable_third_ending=True)
        )
        third = third_generator.generate_run(20260831)
        self.assertEqual([item.floor for item in normal], [1, 2, 3, 4, 5])
        self.assertEqual([item.floor for item in third], [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            [item.fingerprint for item in normal],
            [item.fingerprint for item in third[:5]],
        )

        sixth = third[-1]
        third_generator.validate(sixth)
        self.assertEqual((sixth.width, sixth.height), (6, 5))
        self.assertEqual(len(sixth.nodes), 14)
        self.assertEqual(len(sixth.edges), 17)
        self.assertEqual(
            (sixth.node(sixth.start_node_id).row, sixth.node(sixth.start_node_id).col),
            (2, 0),
        )
        by_position = {
            (node.row, node.col): node for node in sixth.nodes
        }
        self.assertIs(by_position[(2, 3)].node_type, NodeType.BATTLE_BOSS)
        self.assertIs(by_position[(2, 1)].node_type, NodeType.STORY)
        self.assertEqual(by_position[(2, 1)].event_name, "调谐仪式")
        self.assertIs(by_position[(0, 3)].node_type, NodeType.INCIDENT)
        self.assertIs(by_position[(4, 3)].node_type, NodeType.BATTLE_NORMAL)
        self.assertIs(by_position[(2, 5)].node_type, NodeType.BATTLE_SHOP)

    def test_sixth_floor_keeps_observed_slots_and_varies_unspecified_slots(self) -> None:
        assignments = set()
        for seed in range(20):
            floor_map = self.generator.generate_floor(6, seed)
            self.generator.validate(floor_map)
            assignments.add(
                tuple(
                    (node.row, node.col, node.node_type)
                    for node in floor_map.nodes
                )
            )
            by_position = {
                (node.row, node.col): node.node_type for node in floor_map.nodes
            }
            self.assertIs(by_position[(2, 1)], NodeType.STORY)
            self.assertIs(by_position[(0, 3)], NodeType.INCIDENT)
            self.assertIs(by_position[(4, 3)], NodeType.BATTLE_NORMAL)
            self.assertIs(by_position[(2, 5)], NodeType.BATTLE_SHOP)
            self.assertIs(by_position[(2, 3)], NodeType.BATTLE_BOSS)
        self.assertGreater(len(assignments), 1)

    def test_every_verified_template_can_be_filled_without_changing_topology(self) -> None:
        for index, template in enumerate(self.generator.templates.templates):
            floor_map = self.generator.generate_floor(
                template.floor,
                10_000 + index,
                template_id=template.template_id,
            )
            self.assertTrue(template.topology_matches(floor_map), template.template_id)

    def test_rule_distance_shortcuts_across_final_without_mutating_graph(self) -> None:
        edges = {
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
        }
        self.assertEqual(_distances((0, 0), edges)[(1, 2)], 3)
        self.assertEqual(
            _rule_distances((0, 0), edges, {(1, 1)})[(1, 2)],
            2,
        )
        self.assertEqual(len(edges), 3)

    def test_grouped_nodes_and_optional_exits_respect_generation_gates(self) -> None:
        generator = MapGenerator(
            config=MapGeneratorConfig(
                include_advanced_nodes=True,
                enable_portal=True,
                enable_expedition=True,
                enable_second_ending=True,
                door_pair_probability=1.0,
                evacuation_probability=1.0,
            )
        )
        for index, template in enumerate(generator.templates.templates):
            floor_map = generator.generate_floor(
                template.floor,
                20_000 + index,
                template_id=template.template_id,
            )
            counts = Counter(node.node_type for node in floor_map.nodes)
            self.assertEqual(
                counts[NodeType.DOOR],
                2 if template.floor in (3, 4, 5) else 0,
            )
            self.assertEqual(
                counts[NodeType.STORY],
                1 if template.floor == 6 else 3 if template.floor == 5 else 0,
            )
            self.assertEqual(
                counts[NodeType.EVACUATE],
                1 if template.floor in (1, 2, 4) else 0,
            )

    def test_many_maps_obey_graph_and_rule_invariants(self) -> None:
        for seed in range(30):
            for floor_map in self.generator.generate_run(seed):
                self.generator.validate(floor_map)
                floor_rule = self.generator.ruleset.floor(floor_map.floor)
                self.assertLessEqual(len(floor_map.nodes), self.generator.ruleset.max_nodes)
                self.assertLessEqual(
                    max(node.distance_from_start for node in floor_map.nodes),
                    floor_rule.maximum_graph_distance,
                )
                template = self.generator.templates.find_match(floor_map)
                self.assertIsNotNone(template)
                positions = {
                    node.node_id: (node.row, node.col) for node in floor_map.nodes
                }
                for left, right in floor_map.edges:
                    a, b = positions[left], positions[right]
                    self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

                counts = Counter(node.node_type for node in floor_map.nodes)
                for node_type, count in counts.items():
                    if node_type in {
                        NodeType.START,
                        NodeType.FINAL,
                        NodeType.EVACUATE,
                        NodeType.BATTLE_BOSS,
                    }:
                        continue
                    _minimum, maximum = self.generator.ruleset.node_rules[
                        node_type
                    ].count_range(floor_map.floor)
                    self.assertLessEqual(count, maximum)

                self.assertIn(counts[NodeType.DOOR], {0, 2})
                self.assertEqual(counts[NodeType.STORY], 0)
                by_position = {
                    (node.row, node.col): node.node_type for node in floor_map.nodes
                }
                assert template is not None
                for point, node_type in template.fixed_slot_types.items():
                    self.assertEqual(by_position[point], node_type)
                if floor_map.floor in (1, 2):
                    for node_id in floor_map.adjacency()[floor_map.start_node_id]:
                        self.assertEqual(
                            floor_map.node(node_id).node_type,
                            NodeType.BATTLE_NORMAL,
                        )
                if template.boss_terminal is not None:
                    self.assertEqual(
                        by_position[template.boss_terminal],
                        NodeType.BATTLE_BOSS,
                    )
                else:
                    self.assertEqual(
                        {
                            point
                            for point, node_type in by_position.items()
                            if node_type is NodeType.FINAL
                        },
                        set(template.final_slots),
                    )
                    self.assertTrue(
                        all(
                            (node.row, node.col) not in template.final_slots
                            for node in floor_map.nodes
                            if node.node_type is NodeType.EVACUATE
                        )
                    )

    def test_validator_recomputes_distances_and_exit_set(self) -> None:
        floor_map = self.generator.generate_floor(1, 9876)
        target = next(
            node for node in floor_map.nodes if node.node_type is not NodeType.START
        )
        corrupted_nodes = tuple(
            replace(node, distance_from_start=node.distance_from_start + 1)
            if node.node_id == target.node_id
            else node
            for node in floor_map.nodes
        )
        with self.assertRaisesRegex(MapGenerationError, "BFS"):
            self.generator.validate(replace(floor_map, nodes=corrupted_nodes))
        with self.assertRaisesRegex(MapGenerationError, "exit_node_ids"):
            self.generator.validate(replace(floor_map, exit_node_ids=()))


if __name__ == "__main__":
    unittest.main()
