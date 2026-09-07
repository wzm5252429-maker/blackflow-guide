from __future__ import annotations

from dataclasses import dataclass
import unittest

from blackflow_rl.mcts import MCTSConfig, PUCTMCTS, Transition, UniformEvaluator


@dataclass(frozen=True)
class ToyState:
    name: str


class ToyEnvironment:
    action_size = 3

    def legal_action_ids(self, state: ToyState) -> tuple[int, ...]:
        return {"root": (0, 1), "deep": (0,)}.get(state.name, ())

    def transition(self, state: ToyState, action_id: int) -> Transition[ToyState]:
        if state.name == "root" and action_id == 0:
            return Transition(ToyState("deep"), 0.0, False)
        if state.name == "root" and action_id == 1:
            return Transition(ToyState("done"), 1.0, True)
        if state.name == "deep" and action_id == 0:
            return Transition(ToyState("done"), 5.0, True)
        raise ValueError


class MCTSTests(unittest.TestCase):
    def test_single_player_backup_does_not_flip_sign(self) -> None:
        search = PUCTMCTS(
            ToyEnvironment(),
            UniformEvaluator(),
            MCTSConfig(
                num_simulations=128,
                c_puct=1.2,
                gamma=1.0,
                reward_scale=5.0,
                temperature=0.0,
                seed=9,
            ),
        )
        result = search.search(ToyState("root"))
        self.assertEqual(result.selected_action, 0)
        self.assertGreater(result.visit_counts[0], result.visit_counts[1])
        self.assertEqual(len(result.visit_policy), 3)


if __name__ == "__main__":
    unittest.main()
