"""Detector order changes must not invalidate an otherwise confirmed action."""
from dataclasses import replace
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

from blackflow_live.engine import LiveEngine, observation_key
from blackflow_live.models import LiveObservation, ObservedAction, ObservedNode, PolicyDecision


def observed_map(frame_id='first'):
    nodes = (
        ObservedNode('a', 'START', 0, 0, (10, 100, 30, 30)),
        ObservedNode('b', 'WISH', 0, 1, (110, 100, 30, 30)),
        ObservedNode('c', 'SCRAP_SHOP', 1, 0, (10, 200, 30, 30)),
    )
    actions = tuple(ObservedAction('node:' + n.node_id, n.node_type, 'map_node', n.bbox,
                                    target_node_id=n.node_id, metadata={
                                        'operation': 'move', 'movement_cost': 1,
                                        'source_frame_id': frame_id,
                                    }) for n in nodes[1:])
    return LiveObservation(frame_id, time.time(), 'map', .99, actions=actions,
                           nodes=nodes, edges=(('a', 'b'), ('a', 'c')),
                           resources={'hp': 4, 'max_hp': 4, 'gold': 25},
                           current_node_id='a', floor=1)


def reorder(obs, frame_id='fresh'):
    return replace(obs, frame_id=frame_id, captured_at=time.time(),
                   nodes=tuple(reversed(obs.nodes)),
                   edges=tuple((right, left) for left, right in reversed(obs.edges)),
                   actions=tuple(replace(action, metadata={**action.metadata, 'source_frame_id': frame_id})
                                 for action in reversed(obs.actions)))


class LiveObservationOrderTests(unittest.TestCase):
    def test_node_edge_and_action_permutations_have_same_semantic_key(self):
        first = observed_map()
        self.assertEqual(observation_key(first), observation_key(reorder(first)))

    def test_fresh_reordered_confirmation_clicks_once_then_waits(self):
        first = observed_map()
        fresh = reorder(first)
        frame = SimpleNamespace(frame_id=fresh.frame_id, captured_at=fresh.captured_at)
        runtime = SimpleNamespace(
            policy=SimpleNamespace(select=Mock(return_value=PolicyDecision(first.actions[0], 'test'))),
            observe=Mock(return_value=(fresh, frame)), preview=lambda _: b'preview',
            window_info=lambda _: {}, emergency_stop=lambda: False, click=Mock(),
        )
        engine = LiveEngine(lambda _: runtime)
        engine._runtime = runtime
        engine._step(first, object())
        runtime.click.assert_called_once_with(fresh.actions[1], frame)
        engine._step(reorder(first, 'after-click'), object())
        runtime.click.assert_called_once()
        runtime.policy.select.assert_called_once()
        self.assertIn('等待上一步', engine.status()['message'])

    def test_real_map_resource_and_action_changes_still_require_new_decision(self):
        first = observed_map()
        changed = [
            replace(first, nodes=first.nodes[:-1]),
            replace(first, nodes=first.nodes + (ObservedNode('d', 'WISH', 1, 1, (110, 200, 30, 30)),)),
            replace(first, edges=first.edges[:-1]),
            replace(first, edges=first.edges + (('b', 'c'),)),
            replace(first, resources={**first.resources, 'hp': 3}),
            replace(first, resources={**first.resources, 'hp': None}),
            replace(first, resources={key: value for key, value in first.resources.items() if key != 'hp'}),
            replace(first, current_node_id='b'),
            replace(first, actions=first.actions[:-1]),
        ]
        for field, value in (('node_type', 'BATTLE_NORMAL'), ('row', 2), ('col', 2),
                             ('completed', True), ('revealed', False)):
            changed.append(replace(first, nodes=(replace(first.nodes[0], **{field: value}), *first.nodes[1:])))
        for fields in ({'label': 'different target'}, {'kind': 'item_preview'}, {'enabled': False},
                       {'target_node_id': 'c'}, {'metadata': {'operation': 'move', 'movement_cost': 2}},
                       {'metadata': {'operation': 'purchase', 'price': 8}}):
            changed.append(replace(first, actions=(replace(first.actions[0], **fields), *first.actions[1:])))
        for index, current in enumerate(changed):
            with self.subTest(change=index):
                self.assertNotEqual(observation_key(first), observation_key(reorder(current)))

    def test_duplicate_detections_are_not_silently_deduplicated(self):
        first = observed_map()
        for field in ('nodes', 'edges', 'actions'):
            value = getattr(first, field)
            duplicate = replace(first, **{field: value + (value[0],)})
            with self.subTest(field=field):
                self.assertNotEqual(observation_key(first), observation_key(duplicate))


if __name__ == '__main__':
    unittest.main()
