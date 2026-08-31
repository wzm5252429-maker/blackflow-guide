from __future__ import annotations

from dataclasses import replace
import random
import unittest

from blackflow_rl.domain import (
    EventOption,
    FloorMap,
    GameState,
    MapNode,
    NodeType,
    ResourceDelta,
    ResourceState,
)
from blackflow_rl.mapgen import MapGenerator, MapGeneratorConfig
from blackflow_rl.simulator import BlackflowSimulator, InvalidAction


def tiny_state(event: bool = False) -> GameState:
    middle = MapNode(
        node_id="F1_N01",
        index=1,
        row=0,
        col=1,
        node_type=NodeType.INCIDENT if event else NodeType.BATTLE_NORMAL,
        distance_from_start=1,
        options=(
            EventOption("gain", "获得资源", ResourceDelta(gold=5)),
            EventOption("cost", "购买", ResourceDelta(gold=-20, relics=1)),
        )
        if event
        else (),
        auto_effect=ResourceDelta(gold=4, team_strength=1),
    )
    floor_map = FloorMap(
        floor=1,
        width=3,
        height=1,
        nodes=(
            MapNode("F1_N00", 0, 0, 0, NodeType.START, 0),
            middle,
            MapNode("F1_N02", 2, 0, 2, NodeType.FINAL, 2),
        ),
        edges=(("F1_N00", "F1_N01"), ("F1_N01", "F1_N02")),
        start_node_id="F1_N00",
        exit_node_ids=("F1_N02",),
        seed=1,
    )
    return GameState(
        maps=(floor_map,),
        floor_index=0,
        current_node_id="F1_N00",
        resources=ResourceState(action_points=5),
        completed=frozenset({"F1_N00"}),
        revealed=frozenset({"F1_N00", "F1_N01", "F1_N02"}),
    )


class SimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = BlackflowSimulator()

    def test_battle_is_automatic_victory(self) -> None:
        state = tiny_state(event=False)
        transition = self.env.transition(state, 1)
        self.assertIn("F1_N01", transition.next_state.completed)
        self.assertIsNone(transition.next_state.pending_node_id)
        self.assertGreater(transition.next_state.resources.gold, state.resources.gold)
        self.assertGreater(transition.reward, 0)

    def test_noncombat_choice_uses_legal_mask_and_effect(self) -> None:
        state = tiny_state(event=True)
        arrived = self.env.transition(state, 1).next_state
        self.assertEqual(arrived.pending_node_id, "F1_N01")
        self.assertEqual(self.env.legal_action_ids(arrived), (50,))
        with self.assertRaises(InvalidAction):
            self.env.transition(arrived, 51)
        chosen = self.env.transition(arrived, 50).next_state
        self.assertEqual(chosen.resources.gold, state.resources.gold + 5)
        self.assertIn("F1_N01", chosen.completed)

    def test_transitions_are_deterministic_and_immutable(self) -> None:
        state = self.env.reset(99)
        action = self.env.legal_action_ids(state)[0]
        first = self.env.transition(state, action)
        second = self.env.transition(state, action)
        self.assertEqual(first, second)
        self.assertEqual(state.step_count, 0)

    def test_completed_door_exposes_its_pair_at_zero_movement_cost(self) -> None:
        nodes = (
            MapNode("F3_N00", 0, 0, 0, NodeType.START, 0),
            MapNode("F3_N01", 1, 0, 1, NodeType.DOOR, 1, repeatable=True),
            MapNode("F3_N02", 2, 0, 2, NodeType.EMPTY, 2),
            MapNode("F3_N03", 3, 0, 3, NodeType.DOOR, 3, repeatable=True),
            MapNode("F3_N04", 4, 0, 4, NodeType.BATTLE_BOSS, 4),
        )
        floor_map = FloorMap(
            floor=3,
            width=5,
            height=1,
            nodes=nodes,
            edges=tuple(
                (f"F3_N{index:02d}", f"F3_N{index + 1:02d}")
                for index in range(4)
            ),
            start_node_id="F3_N00",
            exit_node_ids=("F3_N04",),
            seed=3,
        )
        state = GameState(
            maps=(floor_map,),
            floor_index=0,
            current_node_id="F3_N01",
            resources=ResourceState(action_points=5),
            completed=frozenset({"F3_N00", "F3_N01"}),
            revealed=frozenset(node.node_id for node in nodes),
        )
        frontier = self.env.reachable_frontier(state)
        self.assertEqual(frontier["F3_N03"], 0)
        self.assertEqual(frontier["F3_N02"], 1)

    def test_random_episodes_always_terminate(self) -> None:
        for seed in range(40):
            state = self.env.reset(seed)
            rng = random.Random(seed)
            for _ in range(300):
                if state.terminal:
                    break
                legal = self.env.legal_action_ids(state)
                self.assertTrue(legal)
                state = self.env.transition(state, rng.choice(legal)).next_state
            self.assertTrue(state.terminal, seed)

    def test_belief_state_does_not_expose_hidden_exact_type(self) -> None:
        state = self.env.reset(0)
        hidden_battle = next(
            node
            for node in state.floor_map.nodes
            if node.is_battle and node.node_id not in state.revealed and not node.is_exit
        )
        belief = self.env.belief_state(state)
        belief_node = belief.floor_map.node(hidden_battle.node_id)
        self.assertEqual(belief_node.node_type, NodeType.BATTLE_NORMAL)

    def test_future_belief_does_not_depend_on_true_future_maps(self) -> None:
        state = self.env.reset(7)
        alternate = self.env.reset(7007)
        same_observation_different_future = replace(
            state,
            maps=(state.maps[0],) + alternate.maps[1:],
        )
        first = self.env.belief_state(state)
        second = self.env.belief_state(same_observation_different_future)
        self.assertEqual(first.maps[1:], second.maps[1:])

    def test_unentered_payload_does_not_change_belief(self) -> None:
        state = self.env.reset(19)
        target = next(
            node
            for node in state.floor_map.nodes
            if node.node_id in state.revealed
            and node.node_id not in state.completed
            and not node.is_exit
        )
        altered_node = replace(
            target,
            auto_effect=ResourceDelta(gold=999, relics=999),
            options=(
                EventOption("secret", "隐藏的真实选项", ResourceDelta(gold=999)),
            ),
            event_name="隐藏的真实事件",
        )
        altered_floor = replace(
            state.floor_map,
            nodes=tuple(
                altered_node if node.node_id == target.node_id else node
                for node in state.floor_map.nodes
            ),
            fingerprint="",
        )
        altered_state = replace(
            state,
            maps=(altered_floor,) + state.maps[1:],
        )
        expected = self.env.belief_state(state).floor_map.node(target.node_id)
        actual = self.env.belief_state(altered_state).floor_map.node(target.node_id)
        self.assertEqual(expected, actual)

    def test_sixth_floor_is_fully_revealed_on_entry(self) -> None:
        generator = MapGenerator(
            config=MapGeneratorConfig(enable_third_ending=True)
        )
        env = BlackflowSimulator(map_generator=generator)
        sixth = generator.generate_run(606)[-1]
        state = GameState(
            maps=(sixth,),
            floor_index=0,
            current_node_id=sixth.start_node_id,
            resources=ResourceState(action_points=5),
            completed=frozenset({sixth.start_node_id}),
            revealed=frozenset({sixth.start_node_id}),
        )
        refreshed = env._refresh_revealed(state)
        self.assertEqual(
            refreshed.revealed,
            frozenset(node.node_id for node in sixth.nodes),
        )


if __name__ == "__main__":
    unittest.main()
