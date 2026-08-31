from __future__ import annotations

import math
import random
from typing import Mapping

from .domain import ActionKind, GameState, ResourceDelta
from .simulator import BlackflowSimulator


class HeuristicEvaluator:
    """Explainable prior baseline derived from configured resource utility."""

    def __init__(self, simulator: BlackflowSimulator, temperature: float = 2.5) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.simulator = simulator
        self.temperature = temperature

    def evaluate(
        self,
        state: GameState,
        legal_action_ids: tuple[int, ...],
    ) -> tuple[Mapping[int, float], float]:
        if not legal_action_ids:
            return {}, 0.0
        scores = {action_id: self.action_score(state, action_id) for action_id in legal_action_ids}
        maximum = max(scores.values())
        weights = {
            action_id: math.exp((score - maximum) / self.temperature)
            for action_id, score in scores.items()
        }
        total = sum(weights.values())
        priors = {action_id: value / total for action_id, value in weights.items()}
        return priors, 0.0

    def action_score(self, state: GameState, action_id: int) -> float:
        action = self.simulator.decode_action(state, action_id)
        objective = self.simulator.ruleset.objective
        if action.kind is ActionKind.CHOOSE:
            node = state.floor_map.node(state.pending_node_id or "")
            option = node.options[action.option_index or 0]
            effect = option.effect
            if option.battle:
                effect = effect + ResourceDelta(gold=4, tickets=1, team_strength=1)
            after = state.resources.apply(effect)
            return objective.resource_reward(
                state.resources,
                after,
                key_items_added=len(set(option.add_items) - set(state.inventory)),
            )

        node = state.floor_map.node(action.target_node_id or "")
        after_move = state.resources.apply(
            ResourceDelta(action_points=-action.movement_cost)
        )
        score = objective.resource_reward(state.resources, after_move)
        if node.options:
            option_scores = []
            for option in node.options:
                if not option.is_available(after_move, state.inventory):
                    continue
                effect = option.effect
                if option.battle:
                    effect = effect + ResourceDelta(gold=4, tickets=1, team_strength=1)
                after = after_move.apply(effect)
                option_scores.append(
                    objective.resource_reward(
                        after_move,
                        after,
                        key_items_added=len(set(option.add_items) - set(state.inventory)),
                    )
                )
            score += max(option_scores, default=0.0)
        else:
            score += objective.resource_reward(
                after_move, after_move.apply(node.auto_effect)
            )
        if node.is_exit:
            score += objective.floor_clear_bonus
            if state.floor_index + 1 == len(state.maps):
                score += objective.run_clear_bonus
        return score


def choose_random_action(
    simulator: BlackflowSimulator,
    state: GameState,
    rng: random.Random,
) -> int:
    legal = simulator.legal_action_ids(state)
    if not legal:
        raise RuntimeError("non-terminal state has no legal actions")
    return rng.choice(legal)


def choose_heuristic_action(
    evaluator: HeuristicEvaluator,
    state: GameState,
) -> int:
    legal = evaluator.simulator.legal_action_ids(state)
    if not legal:
        raise RuntimeError("non-terminal state has no legal actions")
    return max(legal, key=lambda action_id: (evaluator.action_score(state, action_id), -action_id))
