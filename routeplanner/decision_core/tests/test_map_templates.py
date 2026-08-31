from __future__ import annotations

from collections import Counter, deque
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

from blackflow_rl.domain import FloorMap, MapNode, NodeType
from blackflow_rl.map_templates import (
    DEFAULT_TEMPLATE_PATH,
    EXPECTED_TEMPLATE_COUNTS,
    MapTemplate,
    TemplateValidationError,
    load_template_library,
)


class MapTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = load_template_library()

    def test_catalogue_shape_and_floor_index(self) -> None:
        self.assertEqual(len(self.library.templates), 44)
        self.assertEqual(
            Counter(item.floor for item in self.library.templates),
            Counter(EXPECTED_TEMPLATE_COUNTS),
        )
        self.assertEqual(
            {item.template_id for item in self.library.templates},
            {f"1{letter}" for letter in "abc"}
            | {
                f"{floor}{letter}"
                for floor in range(2, 6)
                for letter in "abcdefghij"
            }
            | {"floor-6-01"},
        )
        for floor, expected in EXPECTED_TEMPLATE_COUNTS.items():
            self.assertEqual(len(self.library.for_floor(floor)), expected)

    def test_sixth_floor_uses_observed_fixed_source_confluence_topology(self) -> None:
        template = self.library.get("floor-6-01")
        self.assertEqual((template.rows, template.cols), (5, 6))
        self.assertEqual(template.start, (2, 0))
        self.assertEqual(template.boss_terminal, (2, 3))
        self.assertEqual(len(template.occupied_slots), 14)
        self.assertEqual(len(template.edges), 17)
        self.assertEqual(
            dict(template.fixed_slot_types),
            {
                (2, 1): NodeType.STORY,
                (0, 3): NodeType.INCIDENT,
                (4, 3): NodeType.BATTLE_NORMAL,
                (2, 5): NodeType.BATTLE_SHOP,
            },
        )

    def test_all_templates_obey_graph_invariants(self) -> None:
        for template in self.library.templates:
            self.assertEqual(len(template.edges), len(set(template.edges)))
            self.assertEqual(tuple(sorted(template.edges)), template.edges)
            self.assertIn(template.start, template.occupied_slots)
            self.assertTrue(set(template.terminal_slots) <= template.occupied_slots)
            self.assertTrue(
                set(template.fixed_slot_types) <= template.occupied_slots
            )

            adjacency = {slot: set() for slot in template.occupied_slots}
            for left, right in template.edges:
                self.assertLess(left, right)
                self.assertEqual(
                    abs(left[0] - right[0]) + abs(left[1] - right[1]), 1
                )
                adjacency[left].add(right)
                adjacency[right].add(left)
                for row, col in (left, right):
                    self.assertIn(row, range(template.rows))
                    self.assertIn(col, range(template.cols))

            reached = {template.start}
            queue = deque((template.start,))
            while queue:
                for neighbor in adjacency[queue.popleft()] - reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
            self.assertEqual(reached, set(template.occupied_slots))

            if template.floor in (1, 2, 4):
                self.assertTrue(template.final_slots)
                self.assertIsNone(template.boss_terminal)
            else:
                self.assertFalse(template.final_slots)
                self.assertIsNotNone(template.boss_terminal)

    def test_templates_are_immutable_and_fallback_weights_are_exposed(self) -> None:
        template = self.library.get("1a")
        with self.assertRaises(FrozenInstanceError):
            template.floor = 2  # type: ignore[misc]
        self.assertIsInstance(template.edges, tuple)
        self.assertIsInstance(template.occupied_slots, frozenset)
        self.assertIsNone(template.known_weight)
        self.assertEqual(template.selection_weight, template.fallback_weight)
        self.assertEqual(
            self.library.selection_weights(1),
            tuple(item.fallback_weight for item in self.library.for_floor(1)),
        )

    def test_find_match_uses_coordinates_edges_start_and_terminals(self) -> None:
        template = self.library.get("5a")
        floor_map = self._floor_map(template, width=10)
        self.assertTrue(template.topology_matches(floor_map))
        self.assertIs(self.library.find_match(floor_map), template)

        first_edge = floor_map.edges[0]
        corrupted = replace(
            floor_map,
            edges=tuple(edge for edge in floor_map.edges if edge != first_edge),
        )
        self.assertFalse(template.topology_matches(corrupted))
        self.assertIsNone(self.library.find_match(corrupted))

    def test_loader_rejects_duplicate_undirected_edge(self) -> None:
        raw = json.loads(DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw["templates"][0]["edges"].append(
            list(reversed(raw["templates"][0]["edges"][0]))
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_templates.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(
                TemplateValidationError, "duplicate undirected edge"
            ):
                load_template_library(path)

    @staticmethod
    def _floor_map(template: MapTemplate, *, width: int | None = None) -> FloorMap:
        node_ids = {
            slot: f"r{slot[0]}c{slot[1]}" for slot in template.occupied_slots
        }
        nodes = []
        for index, slot in enumerate(sorted(template.occupied_slots)):
            if slot == template.start:
                node_type = NodeType.START
            elif slot == template.boss_terminal:
                node_type = NodeType.BATTLE_BOSS
            elif slot in template.final_slots:
                node_type = NodeType.FINAL
            else:
                node_type = template.fixed_slot_types.get(slot, NodeType.EMPTY)
            nodes.append(
                MapNode(
                    node_id=node_ids[slot],
                    index=index,
                    row=slot[0],
                    col=slot[1],
                    node_type=node_type,
                    distance_from_start=0,
                )
            )
        return FloorMap(
            floor=template.floor,
            width=template.cols if width is None else width,
            height=template.rows,
            nodes=tuple(nodes),
            edges=tuple((node_ids[left], node_ids[right]) for left, right in template.edges),
            start_node_id=node_ids[template.start],
            exit_node_ids=tuple(node_ids[slot] for slot in template.terminal_slots),
            seed=0,
        )


if __name__ == "__main__":
    unittest.main()
