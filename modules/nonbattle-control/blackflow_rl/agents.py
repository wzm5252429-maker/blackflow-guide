from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import inspect
from pathlib import Path
from typing import Mapping

from .domain import ActionKind, BATTLE_TYPES, GameState, ItemInstance, NodeType, ResourceDelta
from .simulator import BlackflowSimulator
from .future_economy import FutureEconomyValues
from .health_planning import SHADOW_EVENT_NAME, SHADOW_REWARD_ID, shadow_dance_relic_expectation
from .policy_constraints import (DEFAULT_POLICY_CONSTRAINTS, FORBIDDEN_RELIC_IDS, LAKE_OFFERING_CHOICES,
    PolicyConstrainedEnvironment, action_is_allowed, allowed_action_ids, option_is_allowed,
    last_redmoss_tool, redmoss_preparation_active, preparation_cash_target,
    public_node_has_development, public_resident_replaces_node)


def policy_implementation_sha256() -> str:
    """Include every separate policy-value dependency, not just this file."""
    root = Path(__file__).resolve().parent
    names = ("agents.py", "future_economy.py", "health_planning.py", "policy_constraints.py", "duel_rewards.py", "policy_cache.py", "wave_model.py")
    return sha256(b"".join(name.encode()+b"\0"+(root/name).read_bytes() for name in names)).hexdigest()


def unknown_node_distribution(state: GameState, node, ruleset, map_config=None) -> dict[NodeType, float]:
    """An explicit public-information prior, not a recovered hidden node type.

    Current-floor count bounds, known original types and allowed distances
    bound the remaining masses. Unallocated mass follows the generator's
    published research weights. This is a capped marginal approximation to
    its constraint solver, not an exact posterior or a server probability.
    Completed ordinary nodes lose their original type; their EMPTY display
    therefore weakens count minima instead of falsely exhausting EMPTY quota.
    Only public coarse battle/nonbattle labels are read for unrevealed nodes.
    EMPTY is always recognizable on an entered map, never a mystery outcome.
    """
    if (node.node_type is NodeType.EMPTY or node.node_id in state.revealed
            or node.node_id in state.completed):
        return {node.node_type: 1.0}
    portal = getattr(state, "portal_context", None)
    if portal is not None:
        from .portal import BLUE_NODE_TYPES
        variation = portal.variation_id
        pool = ((NodeType.DUEL,) if variation == 8 else
                (NodeType.SACRIFICE,) if variation == 9 else
                (NodeType.EMPTY, NodeType.BATTLE_SAVAGE) if variation == 6 else
                BLUE_NODE_TYPES if variation == 5 else
                (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE))
        # Gold portals share one battle type. A revealed battle makes that
        # public correlation usable, without reading any unrevealed payload.
        if variation == 7:
            known = {other.node_type for other in state.floor_map.nodes
                     if other.node_id in state.revealed and other.node_type in pool}
            if len(known) == 1:
                pool = tuple(known)
        compatible = [kind for kind in pool if kind is not NodeType.EMPTY
                      and (kind in BATTLE_TYPES) == node.is_battle]
        return {kind: 1.0 / len(compatible) for kind in compatible}
    from .mapgen import MapGeneratorConfig, _SPECIAL_FALLBACK_WEIGHTS
    config = map_config or MapGeneratorConfig()
    rules = getattr(ruleset, "node_rules", {})
    unknown = [other for other in state.floor_map.nodes
               if other.node_id not in state.revealed and other.node_id not in state.completed
               and other.node_type is not NodeType.EMPTY
               and other.node_id != state.floor_map.start_node_id and other.is_battle == node.is_battle]
    if not unknown or not rules:
        return {}
    known = Counter()
    ambiguous_completed = 0
    for other in state.floor_map.nodes:
        if other.node_id in state.completed and other.node_type == NodeType.EMPTY:
            ambiguous_completed += 1
        elif (other.node_type is NodeType.EMPTY or other.node_id in state.revealed
              or other.node_id in state.completed):
            known[other.node_type] += 1
    capacities, masses, weights, placements = {}, {}, {}, {}
    for kind, rule in rules.items():
        if (kind in BATTLE_TYPES) != node.is_battle or kind in {NodeType.START, NodeType.EMPTY, NodeType.STORY_HIDDEN}:
            continue
        low, high = rule.count_range(state.floor)
        if kind == NodeType.STORY:
            low = high = 1 if state.floor == 6 else 3 if state.floor == 5 and config.enable_second_ending else 0
        if (kind == NodeType.PORTAL and not config.enable_portal or
                kind == NodeType.EXPEDITION and not config.enable_expedition or
                kind == NodeType.BATTLE_SAVAGE and not config.include_advanced_nodes or
                kind == NodeType.DOOR and config.door_pair_probability == 0 or
                kind == NodeType.EVACUATE and config.evacuation_probability == 0):
            low = high = 0
        allowed = [other.node_id for other in unknown if rule.allows(state.floor, other.distance_from_start)]
        upper = min(len(allowed), max(0, high-known[kind]))
        if upper <= 0:
            continue
        placements[kind] = set(allowed)
        capacities[kind] = float(upper)
        masses[kind] = float(min(upper, max(0, low-known[kind]-ambiguous_completed)))
        weights[kind] = rule.weight if rule.weight > 0 else _SPECIAL_FALLBACK_WEIGHTS.get(kind, 0.1)
        if kind == NodeType.DOOR:
            weights[kind] *= config.door_pair_probability
        elif kind == NodeType.EVACUATE:
            weights[kind] *= config.evacuation_probability
    total_slots = float(len(unknown))
    minimum_mass = sum(masses.values())
    if minimum_mass > total_slots:
        # A partial/custom observation may omit old original node types.
        masses = {kind: mass*total_slots/minimum_mass for kind, mass in masses.items()}
    remaining = max(0.0, total_slots-sum(masses.values()))
    for _ in range(len(masses)+1):
        active = [kind for kind in masses if capacities[kind]-masses[kind] > 1e-9]
        weight_sum = sum(weights[kind] for kind in active)
        if remaining <= 1e-9 or weight_sum <= 0:
            break
        added = {kind: min(capacities[kind]-masses[kind], remaining*weights[kind]/weight_sum) for kind in active}
        for kind, value in added.items():
            masses[kind] += value
        remaining -= sum(added.values())
    propensity = {kind: mass/len(placements[kind]) for kind, mass in masses.items()
                  if node.node_id in placements[kind] and mass > 0}
    total = sum(propensity.values())
    return {kind: value/total for kind, value in propensity.items()} if total else {}


class HeuristicEvaluator:
    """Explainable prior baseline derived from configured resource utility."""

    def __init__(self, simulator: BlackflowSimulator, temperature: float = 2.5) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.simulator = simulator
        self.temperature = temperature
        self.policy_constraints = DEFAULT_POLICY_CONSTRAINTS
        self._last_state: GameState | None = None
        self._last_observation: GameState | None = None

    def observable_state(self, state: GameState) -> GameState:
        if self._last_state is not state:
            self._last_state = state
            self._last_observation = self.simulator.belief_state(state)
        assert self._last_observation is not None
        return self._last_observation

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
        # This baseline is also used directly, outside MCTS. Never price an
        # unentered shop or event from its hidden episode payload.
        state = self.observable_state(state)
        action = self.simulator.decode_action(state, action_id)
        objective = self.simulator.ruleset.objective
        if action.kind is not ActionKind.MOVE:
            option = self.simulator.available_options(state)[action.option_index or 0]
            if action.kind is ActionKind.EQUIP:
                return -0.01
            if action.kind is ActionKind.DISCARD:
                return -1.0
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


@dataclass(frozen=True, slots=True)
class EconomyPolicyConfig:
    """Policy preferences only: none of these values change game rewards."""

    relic_value: float = 24.0
    # Early route opportunity preference only; received relics and P05 stay full value.
    early_wish_value_weight: float = 1.0
    early_gold_value: float = 1.50
    late_gold_value: float = 0.35
    action_point_value: float = 2.2
    shop_value: float = 18.0
    minimum_refresh_cash: int = 12
    green_minimum_refresh_cash: int = 8
    equipment_switch_margin: float = 0.5
    early_cash_reserve: int = 16
    exit_reserve_penalty: float = 7.0
    exit_distance_penalty: float = 1.0
    concept_stacking_discount: float = 0.0
    green_shop_bonus: float = 12.0
    concept_route_weight: float = 0.80
    concept_horizon_weight: float = 1.0
    # Early capital preference only; the final two floors value one-shot
    # concepts at their full one-relic return again.
    one_shot_concept_value_weight: float = 1.0
    shop_reentry_value_weight: float = 1.0
    sell_for_capacity_upgrade: bool = True
    early_relic_delay_floors: int = 0
    early_p06_reserve_target: int = 1
    reveal_node_value: float = 0.0
    activation_mobility_weight: float = 1.0
    shop_liquidity_weight: float = 1.0
    unobserved_relic_price_estimate: float = 16.0
    unknown_node_value_weight: float = 1.0
    vine_future_parts_per_floor: float = 6.0
    vine_trade_priority: float = 45.0
    marked_natural_value_weight: float = 1.0
    sacrifice_conversion_value_weight: float = 1.0
    exchange_downside_weight: float = 0.25
    future_target_access_probability: float = 0.6
    exit_vehicle_value_weight: float = 1.0
    marked_future_reach_probability: float = 0.6
    marked_future_liquidation_probability: float = 0.7
    future_shop_access_probability: float = 0.6
    # Event occurrence is not known; this is an explicit preparation preference.
    shadow_encounter_probability: float = 0.6
    redmoss_tool_keep_value: float = 45.0
    preparation_cash_shortfall_penalty: float = 8.0
    late_exchange_preference: float = 2.0
    # Candidate strategy preferences; defaults keep the old teacher as control.
    wave_growth_weight: float = 0.0
    wave_future_moves_per_floor: float = 6.0

    def __setstate__(self, values):
        # Historical frozen/slotted teacher snapshots contain positional lists.
        # Appended strategy settings must default to the original control.
        from dataclasses import fields
        definitions = fields(self)
        if len(values) > len(definitions):
            raise ValueError('policy configuration snapshot has too many fields')
        for index, definition in enumerate(definitions):
            object.__setattr__(self, definition.name,
                values[index] if index < len(values) else definition.default)


@dataclass(frozen=True, slots=True)
class RelicSearchObjective:
    """A collection task objective, separate from game resource generation."""

    name: str = "relic_collection_v1"
    relic_weight: float = 12.0
    gold_weight: float = 0.05
    movement_use_weight: float = 0.20
    passive_weight: float = 0.20
    chase_penalty: float = -1.5
    defeat_penalty: float = -100.0
    completion_bonus: float = 2.0
    equip_cost: float = 0.01
    external_bank_ingot_penalty: float = 0.02
    # Retained for old objective payload compatibility; unused AP is not a loss.
    fifth_floor_early_exit_ap_penalty: float = 0.0
    discount: float = 0.995
    value_scale: float = 1024.0

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode() + inspect.getsource(type(self).reward).encode()
        return sha256(payload).hexdigest()

    def reward(self, before: GameState, after: GameState, action_kind: ActionKind) -> float:
        def counts(state: GameState) -> tuple[int, int, int]:
            return (
                sum(item.category == "RELIC" for item in state.item_instances),
                sum(item.uses_remaining or 0 for item in state.item_instances if item.category == "MOVE"),
                sum(item.category == "PASSIVE" for item in state.item_instances),
            )
        old, new = counts(before), counts(after)
        reward = self.relic_weight * (new[0] - old[0])
        external_bank = sum(entry.gold_delta for entry in after.ledger[len(before.ledger):] if entry.operation == "bank_withdrawal")
        reward += self.gold_weight * (after.resources.gold - before.resources.gold-external_bank)
        reward -= self.external_bank_ingot_penalty * external_bank
        reward += self.movement_use_weight * (new[1] - old[1])
        reward += self.passive_weight * (new[2] - old[2])
        exhaust_fifth = DEFAULT_POLICY_CONSTRAINTS.fifth_floor_exhaust_actions and before.floor == 5 and before.portal_context is None
        reward += (0.0 if exhaust_fifth else self.chase_penalty) * (after.chase_count - before.chase_count)
        if before.resources.hp > 0 and after.resources.hp <= 0:
            reward += self.defeat_penalty
        if after.terminal and not before.terminal and after.resources.hp > 0:
            reward += self.completion_bonus
        if action_kind == ActionKind.EQUIP:
            reward -= self.equip_cost
        return reward


class RelicSearchEnvironment(PolicyConstrainedEnvironment):
    """PUCT adapter: transitions are unchanged, only search rewards differ."""

    def __init__(self, simulator: BlackflowSimulator, objective: RelicSearchObjective | None = None, *, constraints=DEFAULT_POLICY_CONSTRAINTS):
        super().__init__(simulator, constraints=constraints)
        self.objective = objective or RelicSearchObjective()

    @property
    def action_size(self) -> int:
        return self.simulator.action_size

    def legal_action_ids(self, state: GameState) -> tuple[int, ...]:
        return allowed_action_ids(self.simulator, state, self.policy_constraints)

    def transition(self, state: GameState, action_id: int):
        action = self.simulator.decode_action(state, action_id)
        outcome = super().transition(state, action_id)
        if not state.economy_enabled:
            return outcome
        reward = self.objective.reward(state, outcome.next_state, action.kind)
        return replace(outcome, reward=reward, info={**outcome.info,
            "resource_reward": outcome.reward, "search_objective": self.objective.name})


class EconomyEvaluator(FutureEconomyValues, HeuristicEvaluator):
    """Observable inventory, shopping and movement baseline for relic collecting.

    Buy/sell decisions use the same public quotes as the shop UI. Only equip
    actions (which reveal no new node outcome) are previewed on the live state.
    Unentered node payloads and future random shop refreshes are never inspected.
    """

    def __init__(self, simulator: BlackflowSimulator, config: EconomyPolicyConfig | None = None) -> None:
        super().__init__(simulator, temperature=4.0)
        self.config = config or EconomyPolicyConfig()

    def _gold_value(self, state: GameState) -> float:
        fraction = state.floor_index / max(1, state.floor_index + self._remaining_floors(state) - 1)
        return self.config.early_gold_value * (1.0 - fraction) + self.config.late_gold_value * fraction

    def _health_effect_value(self, state: GameState, effect: ResourceDelta) -> float:
        after = state.resources.apply(effect)
        ordinary = (after.hp-state.resources.hp)*0.05 + (after.max_hp-state.resources.max_hp)*0.1
        if SHADOW_EVENT_NAME in state.seen_event_names:
            return ordinary
        pending = state.floor_map.node(state.pending_node_id) if state.pending_node_id else None
        active = pending is not None and pending.event_name == SHADOW_EVENT_NAME
        stage = min(7, dict(state.event_counters).get(state.pending_node_id+":stage", 0)) if active else 0
        weight = 1.0 if active else self.config.shadow_encounter_probability
        if not active and state.floor >= 5 and state.portal_context is None and state.resources.action_points <= 0:
            return ordinary
        had = SHADOW_REWARD_ID in state.inventory
        before_value = shadow_dance_relic_expectation(state.resources.hp, state.resources.max_hp,
            had_reward_on_entry=had, completed_circles=stage)
        after_value = shadow_dance_relic_expectation(after.hp, after.max_hp,
            had_reward_on_entry=had, completed_circles=stage)
        return ordinary + weight*self.config.relic_value*(after_value-before_value)

    def _max_hp_value(self, state: GameState, amount: int) -> float:
        return self._health_effect_value(state, ResourceDelta(max_hp=amount))

    def _travel_health_value(self,state,node,lost,*,action_points_after):
        """Value path damage before the actual arrival event is resolved.

        A final-AP unknown event can still be the shadow itself. A known
        terminal plain square cannot preserve a fictitious later encounter.
        """
        residents=state.resident_context
        occupied=bool(residents and (node.node_id in residents.occupied_node_ids or
            node.node_id in residents.stronghold_node_ids-residents.defeated_stronghold_ids))
        current_incident=(node.node_id not in state.completed and not occupied and
            (node.node_type==NodeType.INCIDENT if node.node_id in state.revealed else not node.is_battle))
        ap=max(1,action_points_after) if current_incident else max(0,action_points_after)
        projected=replace(state,resources=replace(state.resources,action_points=ap))
        return self._health_effect_value(projected,ResourceDelta(hp=-lost))

    def _exit_health_value(self, state: GameState, node) -> float:
        config = getattr(self.simulator.economy, "config", None)
        if (state.portal_context is not None or state.floor >= 5 or node.node_type not in {NodeType.FINAL, NodeType.EVACUATE}
                or node.node_id not in state.revealed and node.node_id != state.pending_node_id
                or not getattr(config, "full_tech", False) or getattr(config, "difficulty", 0) < 9):
            return 0.0
        return self._max_hp_value(state, 2)

    def _preparation_cash_penalty(self, state: GameState, cost: int) -> float:
        target = preparation_cash_target(state, self.policy_constraints)
        before = max(0, target-state.resources.gold)
        after = max(0, target-state.resources.gold+cost)
        return max(0, after-before)*self.config.preparation_cash_shortfall_penalty

    def _known_green_departure_cash_target(self, state: GameState) -> int | None:
        """Cash needed for one currently feasible, publicly known green entry.

        This projects only public completion effects and navigation geometry,
        without sampling completion rewards or transition outcomes. A random
        transport is never treated as a controllable destination.
        """
        if (state.floor != 2 or state.portal_context is not None or not state.pending_node_id
                or state.resources.action_points <= 0 or state.terminal
                or state.floor_map.node(state.pending_node_id).node_type != NodeType.BATTLE_SHOP):
            return None
        occupied = state.resident_context.occupied_node_ids if state.resident_context else frozenset()
        targets = {node.node_id for node in state.floor_map.nodes if node.node_id in state.revealed
            and node.node_type == NodeType.SCRAP_SHOP and node.node_id not in occupied}
        if not targets:
            return None
        region = state.region_state
        if region is not None and state.current_node_id not in state.completed:
            region = region.clear_source(state.current_node_id)
        navigation = replace(state, pending_node_id=None, region_state=region,
            completed=state.completed | {state.current_node_id})
        destination_capital = getattr(self.policy_constraints, "second_floor_green_entry_gold", 12)
        possible = []
        for gear in (None, *(item for item in state.item_instances
                if item.category == "MOVE" and (item.uses_remaining or 0) > 0)):
            definition = self._definition(gear.item_id) if gear else None
            if definition is not None and definition.random_move:
                continue
            equipped = replace(navigation, equipped_instance_id=gear.instance_id if gear else None)
            for action in self.simulator.legal_actions(equipped):
                if action.kind != ActionKind.MOVE or action.target_node_id not in targets:
                    continue
                trajectory = (action.target_node_id,) if gear else action.traversed_node_ids
                region = navigation.region_state
                fee = (2*sum(region.affects(node_id) for node_id in trajectory)
                    if region is not None and region.active and region.policy in {"改良", "improve"} else 0)
                earned = 4 if gear and gear.item_id.endswith("M_10") else 0
                destination=replace(navigation,current_node_id=action.target_node_id,
                    item_instances=tuple(replace(item,appraisal=max(0,item.appraisal-2))
                        if item.item_id=='rogue_6_scrap_G_10' else item for item in state.item_instances))
                goods=sum(self._sell_quote(destination,item) for item in destination.item_instances
                    if item.category=='GOODS' and not last_redmoss_tool(destination,item,self.policy_constraints))
                destination_cash=max(0,destination_capital-goods)
                # Runtime first clamps the path debit at zero, then pays M10.
                needed = 0 if earned >= destination_cash else fee+destination_cash-earned
                possible.append(int(needed))
        return min(possible) if possible else None

    @staticmethod
    def _project_goods_sale(state: GameState, instance: ItemInstance, proceeds: float) -> GameState:
        remaining_items = tuple(item for item in state.item_instances if item.instance_id != instance.instance_id)
        return replace(state, item_instances=remaining_items,
            inventory=(state.inventory if any(item.item_id == instance.item_id for item in remaining_items)
                else state.inventory - {instance.item_id}),
            equipped_instance_id=None if state.equipped_instance_id == instance.instance_id else state.equipped_instance_id,
            resources=replace(state.resources, gold=state.resources.gold+proceeds,
                parts=max(0, state.resources.parts-1)))

    def _known_green_sale_preparation_value(self, state: GameState, instance: ItemInstance | None,
            proceeds: float, score: float) -> float:
        if (instance is None or instance.category == "RELIC" or instance.item_id.endswith("G_09")
                or last_redmoss_tool(state, instance, self.policy_constraints)):
            return score
        goal = self._known_green_departure_cash_target(state)
        if goal is None or state.resources.gold >= goal:
            return score
        funded = self._project_goods_sale(state, instance, proceeds)
        after_goal = self._known_green_departure_cash_target(funded)
        if after_goal is None:
            return score  # Selling the only usable transport cannot fund its route.
        reduction = max(0.0, goal-state.resources.gold-max(0.0, after_goal-funded.resources.gold))
        return score+reduction*self.config.preparation_cash_shortfall_penalty

    def _green_entry_preparation_penalty(self, state: GameState, node, gold_after_path: float) -> float:
        """Latest entry budget accepts cash or real saleable natural assets."""
        if state.floor != 2 or state.portal_context is not None or node.node_type != NodeType.SCRAP_SHOP:
            return 0.0
        target = getattr(self.policy_constraints, "second_floor_green_entry_gold", 12)
        destination = replace(state, current_node_id=node.node_id)
        saleable = sum(self._sell_quote(destination, item) for item in state.item_instances
            if item.category == "GOODS" and not last_redmoss_tool(state, item, self.policy_constraints))
        deficit=max(0.0,target-gold_after_path-saleable)
        return min(self.config.relic_value,deficit*self.config.preparation_cash_shortfall_penalty)

    def _has_future_navigation(self, state: GameState) -> bool:
        if state.terminal:
            return False
        if state.floor < 5 or state.resources.action_points > 0 or state.pending_node_id is None:
            return True
        if state.portal_context is not None and state.portal_context.outer_action_points > 0:
            return True
        node = state.floor_map.node(state.pending_node_id)
        if node.event_name == SHADOW_EVENT_NAME:
            return True  # An actual later AP circle can reopen movement.
        for option in node.options:
            if not option_is_allowed(state, option, self.policy_constraints):
                continue
            if option.effect.action_points > 0:
                return True
            definition = self._definition(option.item_id)
            if definition and definition.category == "RELIC" and option.operation in {"take", "purchase", "event_reward"}:
                from .relic_effects import blackboard
                if any(buff["key"] == "immediate_reward" and blackboard(buff).get("id") == "rogue_6_ap" for buff in definition.client_buffs):
                    return True
        return False

    def _remaining_floors(self, state: GameState) -> int:
        return len(state.maps) - state.floor_index + len(self._promised_floors(state))

    def _one_shot_concept_weight(self, state: GameState) -> float:
        return self.config.one_shot_concept_value_weight if self._remaining_floors(state) > 2 else 1.0

    def _promised_floors(self, state: GameState) -> tuple[int, ...]:
        result = []
        promised_beacon = not getattr(self.policy_constraints, "first_ending_only", True) and ("rogue_6_relic_final_3" in state.inventory or any(request.kind == "source" for request in getattr(state, "expeditions", ())))
        if promised_beacon and not any(floor.floor == 6 for floor in state.maps):
            result.append(6)
        if "rogue_6_relic_artifact_5" in state.inventory and not dict(state.event_counters).get("remembrance:spent", 0) and state.floor <= 4:
            result.append(4)
        return tuple(result)

    def _needs_source_expedition(self, state: GameState) -> bool:
        return not getattr(self.policy_constraints, "first_ending_only", True) and "rogue_6_relic_final_3" not in state.inventory and not any(request.kind == "source" for request in getattr(state, "expeditions", ()))

    def _definition(self, item_id: str | None):
        economy = getattr(self.simulator, "economy", None)
        catalog = getattr(economy, "catalog", None)
        return catalog.items.get(item_id) if catalog is not None and item_id else None

    def _sell_quote(self, state: GameState, instance: ItemInstance) -> float:
        economy = getattr(self.simulator, "economy", None)
        quote = getattr(economy, "quote_sell", None)
        if quote is not None:
            return float(quote(state, instance))
        definition = self._definition(instance.item_id)
        return float((definition.sell_price or 0) + instance.appraisal) if definition else 0.0

    def _pool_value(self, state: GameState, item_id: str) -> float:
        name = item_id.removeprefix("pool:")
        if name == "rogue_6:狭路相逢：中":
            from .duel_rewards import expected_duel_middle_value
            return expected_duel_middle_value(lambda candidate: self._new_part_reward_value(state, self._definition(candidate)))
        category = name.split(":")[0]
        economy = self.simulator.economy
        catalog = economy.catalog
        if category == "RELIC":
            if hasattr(economy, "candidates"):
                rarity = (name.split(":", 1)[1],) if ":" in name else None
                if not economy.candidates(state, "RELIC", rarity=rarity):
                    return 0.0
            return self.config.relic_value
        if category in {"PART", "MOVE", "GOODS", "PASSIVE"}:
            return 5.0+self._acquisition_growth_cash(state)*self._gold_value(state)
        return sum((self.config.relic_value if catalog.items[candidate].category == "RELIC" else
            self._new_part_reward_value(state, catalog.items[candidate])) * probability
            for candidate, probability in self._pool_distribution(state, name).items())

    def _new_part_reward_value(self, state: GameState, definition) -> float:
        if definition is None:
            return 0.0
        value = self._part_future_value(state, definition)
        quantity = 4 if definition.item_id.endswith("G_07") else 1
        if quantity == 4:
            # The awarded G07 receives growth from its three real moss balls;
            # neither the new item nor those balls exist in `state` yet.
            value += 3*self._gold_value(state)
            moss = self._definition("rogue_6_scrap_G_08")
            if moss:
                value += 3*self._sell_quote(state, ItemInstance("ball_reward_quote", moss.item_id, "GOODS", appraisal=int(moss.sell_price or 0)))*self._gold_value(state)
        return value+self._acquisition_growth_cash(state, quantity)*self._gold_value(state)

    def _pool_distribution(self, state: GameState, name: str) -> dict[str, float]:
        economy = self.simulator.economy
        catalog = economy.catalog
        if not hasattr(catalog, "observed_weights") or name not in catalog.observed_pools:
            return {}
        from .economic_sampling import pool_members
        difficulty = economy.config.difficulty
        available = {candidate for kind in ("RELIC", "MOVE", "GOODS", "PASSIVE")
            for candidate in economy.candidates(state, kind)}
        available.intersection_update(pool_members(catalog, name, difficulty=difficulty))
        if not available:
            return {}
        weights: dict[str, int] = {}
        for candidate, weight in catalog.observed_weights(name):
            if candidate not in catalog.items:
                continue
            resolved = catalog.variant(candidate, difficulty=difficulty).item_id
            if resolved in available:
                weights[resolved] = weights.get(resolved, 0)+weight
        # Same explicit uniform known-member fallback as engine sampling.
        if not weights:
            weights = {candidate: 1 for candidate in available}
        total = sum(weights.values())
        return {candidate: weight/total for candidate, weight in weights.items()}

    def _relic_bonus_value(self, state: GameState, definition) -> float:
        if definition is None:
            return 0.0
        from .relic_effects import blackboard
        remaining_floors = self._remaining_floors(state)
        value = self._relic_future_bonus(state, definition)
        if definition.item_id == "rogue_6_relic_final_3" and self._needs_source_expedition(state):
            value += self.config.relic_value * 8
        if definition.item_id == "rogue_6_relic_artifact_5" and state.floor <= 4 and not dict(state.event_counters).get("remembrance:spent", 0):
            value += self.config.relic_value * 8
        seen_discount = False
        health_effect = ResourceDelta()
        for buff in definition.client_buffs:
            bb = blackboard(buff)
            key = buff["key"]
            target = bb.get("id")
            count = float(bb.get("count", 1))
            if key == "immediate_reward":
                if target == "rogue_6_gold":
                    value += count * self._gold_value(state)
                elif target == "rogue_6_ap":
                    value += count * self.config.action_point_value
                elif target in {"rogue_6_hp", "rogue_6_hpmax"}:
                    health_effect += ResourceDelta(**{"hp" if target == "rogue_6_hp" else "max_hp": int(count)})
                elif target in self.simulator.economy.catalog.observed_pools:
                    value += count * self._pool_value(state, "pool:" + target)
                elif self._definition(target) is not None:
                    granted = self._definition(target)
                    value += count * (self.config.relic_value if granted.category == "RELIC" else self._part_future_value(state, granted))
            elif key == "shop_discount_item" and not seen_discount:
                seen_discount = True
                spending = state.resources.gold + max(0, remaining_floors - 1) * 20
                value += min(100.0, spending) * (1.0 - float(bb.get("ratio", 1))) * self._gold_value(state)
            elif key == "immediate_reward_on_vehicle_broken":
                value += min(10, remaining_floors * 3) * 5.0
            elif key == "player_level_rewards" and target == "rogue_6_gold":
                commander_level = dict(state.event_counters).get("commander_level", 1)
                likelihood = min(1.0, (commander_level + remaining_floors * 1.8) / max(1, float(bb.get("level", 10))))
                value += count * self._gold_value(state) * likelihood
            elif key == "scrap_fill_up":
                value += max(0, state.parts_capacity - state.resources.parts) * self._part_future_value(state, self._definition(target))
        if definition.item_id == "rogue_6_relic_cargo_6":
            value += remaining_floors * 4 * self.config.action_point_value
        return value+self._health_effect_value(state, health_effect)

    def _part_future_value(self, state: GameState, definition, *, move_uses: int | None = None) -> float:
        if definition is None:
            return 0.0
        if definition.category in {"MOVE", "PASSIVE"} and not self._has_future_navigation(state):
            return 0.0
        remaining_floors = self._remaining_floors(state)
        if definition.category == "GOODS":
            sale = float(definition.sell_price or 0)
            protection = self.config.redmoss_tool_keep_value if (definition.item_id.endswith(("G_01", "G_12"))
                and redmoss_preparation_active(state, self.policy_constraints)
                and not any(item.item_id.endswith(("G_01", "G_12")) for item in state.item_instances)) else 0.0
            if definition.item_id.endswith("G_02"):
                sale += min(40, remaining_floors * 4)
            elif definition.item_id.endswith("G_03"):
                sale += min(60, remaining_floors * 8)
            elif definition.item_id.endswith("G_04"):
                sale += min(60, remaining_floors * 6)
            elif definition.item_id.endswith("G_07"):
                # Spawned moss balls are valued separately at purchase time.
                sale += min(40, remaining_floors * 6)
            elif definition.item_id.endswith("G_09"):
                sale += 4*self._vine_growth_opportunities(state)
            elif definition.item_id.endswith("G_05"):
                sale += self._wave_growth(state, int(sale))
            return max(protection, sale * self._gold_value(state))
        if definition.category == "MOVE":
            existing_uses = sum(item.uses_remaining or 0 for item in state.item_instances if item.category == "MOVE")
            scarcity = 1.0 / (1.0 + existing_uses / 3.0)
            extent = max((abs(row) + abs(col) for row, col in definition.move_range),
                default=max(state.floor_map.width, state.floor_map.height))
            uses = min((definition.move_uses or 1) if move_uses is None else move_uses, remaining_floors * 3)
            if uses <= 0:
                return 0.0
            movement_value = scarcity * uses * (1.5 + min(extent, 6) * 0.55)
            if definition.move_ap == 0:
                movement_value += uses * self.config.action_point_value
            elif "rogue_6_relic_cargo_6" in state.inventory:
                movement_value += self.config.action_point_value
            if definition.item_id.endswith("M_12"):
                movement_value += 3 * self.config.action_point_value
            if definition.item_id.endswith("M_10"):
                movement_value += 4 * self._gold_value(state) * uses
                movement_value += self._shop_reentry_value(state, None) * uses
            movement_value += self._marginal_vehicle_activations(state, definition, uses) * self.config.activation_mobility_weight
            return movement_value
        if definition.category == "PASSIVE":
            if definition.item_id.endswith(("P_01", "P_02")):
                return self.config.relic_value * self._one_shot_concept_weight(state) * (1.0 if any(item.category == "MOVE" for item in state.item_instances) else 0.6)
            if definition.item_id.endswith(("P_05", "P_06")):
                copies = sum(item.item_id == definition.item_id for item in state.item_instances)
                mobility = 1.0 if any(item.category == "MOVE" for item in state.item_instances) else 0.4
                target_type = NodeType.SACRIFICE if definition.item_id.endswith("P_06") else NodeType.WISH
                broad_horizon = min(5, remaining_floors * 1.5)
                observable_horizon = self._concept_opportunities(state, target_type) if self.config.concept_horizon_weight else broad_horizon
                horizon = broad_horizon * (1 - self.config.concept_horizon_weight) + observable_horizon * self.config.concept_horizon_weight
                return self.config.relic_value * horizon * (2 if definition.item_id.endswith("P_06") else 1) * mobility / (1 + copies * self.config.concept_stacking_discount)
            return 0.0
        return 0.0

    def _wave_horizon(self, state: GameState) -> int:
        if not self._has_future_navigation(state):
            return 0
        future = max(0, self._remaining_floors(state)-1)*self.config.wave_future_moves_per_floor
        free_uses = sum(item.uses_remaining or 0 for item in state.item_instances
            if item.category == 'MOVE' and self._definition(item.item_id) is not None
            and self._definition(item.item_id).move_ap == 0)
        return max(0, min(40, int(future+max(0, state.resources.action_points)+free_uses)))

    def _wave_growth(self, state: GameState, appraisal: int, *, moves=None) -> float:
        if not self.config.wave_growth_weight:
            return 0.0
        from .wave_model import expected_appraisal
        tail = self.simulator.economy.config.synthetic_wave_tail_probability
        horizon = self._wave_horizon(state) if moves is None else moves
        value = max(0, min(999, appraisal))
        return (expected_appraisal(value, horizon, tail)-value)*self.config.wave_growth_weight

    def _vine_growth_opportunities(self, state: GameState) -> float:
        # The former six acquisitions/floor prior is now explicit. Current
        # floor opportunities depend on remaining AP and known unsold parts;
        # future floors use that fixed policy prior, not private generated maps.
        future = max(0, self._remaining_floors(state)-1)*self.config.vine_future_parts_per_floor
        current = min(self.config.vine_future_parts_per_floor, max(0, state.resources.action_points)*0.75)
        if state.pending_node_id:
            shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
            if shop is not None:
                current += sum(not slot.sold and not slot.item_id.endswith("G_09") and self._definition(slot.item_id) is not None
                    and self._definition(slot.item_id).category in {"MOVE", "GOODS", "PASSIVE"} for slot in shop.stock)
        return min(30.0, future+current)

    def _acquisition_growth_cash(self, state: GameState, quantity: int = 1) -> float:
        """Resalable appraisal added to already-held natural objects only."""
        return sum(self._sell_quote(state, replace(item, appraisal=item.appraisal+
                    quantity*(4 if item.item_id.endswith("G_09") else 1)))-self._sell_quote(state, item)
            for item in state.item_instances if item.item_id.endswith(("G_07", "G_09")))

    def _shop_roundtrip_margin(self, state: GameState, definition, cost: float) -> float | None:
        if definition is None or definition.category not in {"MOVE", "GOODS", "PASSIVE"} or definition.item_id.endswith("G_09"):
            return None
        node = state.floor_map.node(state.pending_node_id) if state.pending_node_id else None
        config = getattr(self.simulator.economy, "config", None)
        if node is None or not (node.node_type == NodeType.SCRAP_SHOP or node.node_type == NodeType.BATTLE_SHOP and getattr(config, "full_tech", False)):
            return None
        if state.resources.parts > state.parts_capacity:
            return None
        instance = ItemInstance("roundtrip_quote", definition.item_id, definition.category,
            definition.move_uses or definition.passive_uses, int(definition.sell_price or 0))
        # A purchase does not repeat a three-sale bonus. Its marginal third
        # sale is included only when exactly two actual sales preceded it.
        shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
        if definition.item_id.endswith("G_07"):
            moss = self._definition("rogue_6_scrap_G_08")
            if state.resources.parts+1 > state.parts_capacity or moss is None:
                return None
            grown = replace(instance, appraisal=instance.appraisal+3)
            balls = 3*self._sell_quote(state, ItemInstance("ball_quote", moss.item_id, "GOODS", appraisal=int(moss.sell_price or 0)))
            bonus = 8 if shop and not shop.trade_bonus_paid and getattr(config, "squad", None) == "multilateral_trade" else 0
            return self._sell_quote(state, grown)+balls+self._acquisition_growth_cash(state, 4)+bonus-cost
        bonus = 8 if shop and shop.sold_parts == 2 and not shop.trade_bonus_paid and getattr(config, "squad", None) == "multilateral_trade" else 0
        return self._sell_quote(state, instance)+self._acquisition_growth_cash(state)+bonus-cost

    def _latest_overflow_purchase(self, state: GameState, instance: ItemInstance) -> bool:
        """A capacity-neutral same-shop round trip must sell its own new item."""
        if state.resources.parts <= state.parts_capacity or instance.item_id.endswith(("G_07", "G_09")):
            return False
        purchases = [entry for entry in state.ledger if entry.operation == "purchase"]
        if not purchases or purchases[-1].step != state.step_count:
            return False
        return any(entry.operation == "acquire" and entry.source == "shop" and entry.instance_id == instance.instance_id
            and entry.step == purchases[-1].step for entry in state.ledger)

    def _held_part_value(self, state: GameState, instance: ItemInstance) -> float:
        definition = self._definition(instance.item_id)
        if instance.category != "MOVE":
            value = self._part_future_value(state, definition)
            if instance.item_id.endswith('G_05') and self.config.wave_growth_weight:
                value = (self._sell_quote(state, instance)+self._wave_growth(state, instance.appraisal))*self._gold_value(state)
            if last_redmoss_tool(state, instance, self.policy_constraints):
                value = max(value, self.config.redmoss_tool_keep_value)
            if instance.item_id.endswith("G_12"):
                known_uses = [node.event_name for node in state.floor_map.nodes
                    if node.node_id in state.revealed and node.node_id not in state.completed]
                if "呼吸的红苔" in known_uses:
                    value = max(value, 30*self._gold_value(state))
                if "愈创之心" in known_uses:
                    value = max(value, 5*self.config.action_point_value)
            return value
        # Its own charges must not be counted as a substitute for keeping it.
        without = replace(state, item_instances=tuple(item for item in state.item_instances if item.instance_id != instance.instance_id),
            resources=replace(state.resources, parts=max(0, state.resources.parts-1)))
        return self._part_future_value(without, definition, move_uses=instance.uses_remaining or 0)

    def _vehicle_target_quality(self, state: GameState, definition, kind: NodeType) -> float:
        if definition is None or definition.category != "MOVE":
            return 0.0
        targets = definition.move_target_types
        battle = kind == NodeType.BATTLE_NORMAL
        allowed = "ALL" in targets or (not battle and "EVENTS" in targets) or kind.value in targets
        if kind == NodeType.BATTLE_NORMAL:
            allowed = allowed or any(name.startswith("BATTLE_") and name != "BATTLE_SHOP" for name in targets)
        elif kind == NodeType.EMPTY:
            allowed = allowed or any(not name.startswith("BATTLE_") or name == "BATTLE_SHOP" for name in targets)
        if not allowed:
            return 0.0
        if definition.random_move:
            return 1.0 if kind == NodeType.EMPTY else 0.15
        if not definition.move_range:
            return 1.0
        origin = state.floor_map.node(state.current_node_id)
        known = [node for node in state.floor_map.nodes if node.node_id in state.revealed and
            node.node_id not in state.completed and node.node_type == kind]
        if known:
            direct = sum((node.row-origin.row, node.col-origin.col) in definition.move_range for node in known) / len(known)
            return 0.25 + 0.75 * direct
        extent = max(abs(row) + abs(col) for row, col in definition.move_range)
        return min(1.0, 0.25 + extent / 6.0)

    def _marginal_vehicle_activations(self, state: GameState, definition, uses: int) -> float:
        """Do not promise the same finite concept targets to every new vehicle.

        Observable target counts plus the existing declared future-floor prior
        form a policy budget. Existing compatible charges use that budget first;
        geometry weights retain value for a long-range upgrade over short hops.
        """
        value = 0.0
        for suffix, kind, multiplier in (("P_01", NodeType.BATTLE_NORMAL, 1),
                ("P_02", NodeType.EMPTY, 1), ("P_05", NodeType.WISH, 1),
                ("P_06", NodeType.SACRIFICE, 2)):
            copies = sum(item.item_id.endswith(suffix) for item in state.item_instances)
            quality = self._vehicle_target_quality(state, definition, kind)
            if not copies or not quality:
                continue
            persistent = suffix in {"P_05", "P_06"}
            opportunities = (self._concept_opportunities(state, kind, include_future=not definition.expires_on_floor_change) if hasattr(self.simulator.ruleset, "node_rules") else
                min(5.0, self._remaining_floors(state) * 1.5)) if persistent else 1.0
            existing = sum((item.uses_remaining or 0) * self._vehicle_target_quality(state, self._definition(item.item_id), kind)
                for item in state.item_instances if item.category == "MOVE")
            marginal = max(0.0, min(opportunities, existing + uses * quality) - min(opportunities, existing))
            future_weight = 1.0 if persistent else self._one_shot_concept_weight(state)
            value += marginal * copies * multiplier * self.config.relic_value * future_weight
        return value

    def _concept_opportunities(self, state: GameState, target_type: NodeType, *, include_future: bool = True) -> float:
        """Remaining public opportunities; future floors use bounded priors.

        Count ranges are constraints, not measured probabilities. Their midpoint
        times a route-access preference is an explicit policy estimate only.
        """

        rules = self.simulator.ruleset.node_rules[target_type]
        adjacency = state.floor_map.adjacency()
        distances = {state.current_node_id: 0}
        queue = deque([state.current_node_id])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        origin = state.floor_map.node(state.current_node_id)
        current_value = 0.0
        for node in state.floor_map.nodes:
            if node.node_id not in state.revealed or node.node_id in state.completed or node.node_type != target_type:
                continue
            distance = distances.get(node.node_id, 999)
            accessible = distance <= max(0, state.resources.action_points - 1)
            for item in state.item_instances:
                definition = self._definition(item.item_id)
                if item.category != "MOVE" or definition is None or definition.random_move:
                    continue
                if definition.move_range and (node.row - origin.row, node.col - origin.col) not in definition.move_range:
                    continue
                targets = definition.move_target_types
                if "ALL" not in targets and "EVENTS" not in targets and target_type.value not in targets:
                    continue
                if int(definition.move_ap or 0) <= state.resources.action_points:
                    accessible = True
            current_value += 0.85 if accessible else 0.15
        unexplored = sum(node.node_id not in state.revealed and not node.is_battle for node in state.floor_map.nodes)
        mystery_total = sum(not node.is_battle for node in state.floor_map.nodes)
        low, high = rules.count_range(state.floor)
        hidden_value = (low + high) * 0.5 * unexplored / max(1, mystery_total) * 0.5
        future_value = sum(sum(rules.count_range(floor.floor)) * 0.5 * 0.6
            for floor in state.maps[state.floor_index + 1:])
        future_value += sum(sum(rules.count_range(floor)) * 0.5 * 0.6 for floor in self._promised_floors(state))
        return current_value + hidden_value + (future_value if include_future else 0.0)

    def action_score(self, state: GameState, action_id: int) -> float:
        state = self.observable_state(state)
        score = self._economic_action_score(state, action_id)
        if not state.economy_enabled:
            return score
        action = self.simulator.decode_action(state, action_id)
        if action.option_index is None:
            return score
        option = self.simulator.available_options(state)[action.option_index]
        instance = next((item for item in state.item_instances if item.instance_id == option.instance_id), None)
        if option.operation in {"sell", "discard", "cultivate", "exchange"} and last_redmoss_tool(state, instance, self.policy_constraints):
            return min(score, -self.config.redmoss_tool_keep_value)
        if option.operation == "sell":
            shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
            proceeds = option.effect.gold if option.effect.gold > 0 else option.price or 0
            if shop and shop.sold_parts == 2 and not shop.trade_bonus_paid:
                proceeds += 8
            score = self._known_green_sale_preparation_value(state, instance, proceeds, score)
        if option.operation in {"purchase", "refresh", "event_reward", "event", "event_advance", "wish_refresh"} and option.option_id not in LAKE_OFFERING_CHOICES:
            score -= self._preparation_cash_penalty(state, max(option.price or 0, -option.effect.gold, 0))
        if option.effect.gold > 0 and option.operation not in {"sell", "bank_withdraw"}:
            deficit = max(0, preparation_cash_target(state, self.policy_constraints)-state.resources.gold)
            score += min(deficit, option.effect.gold)*self._gold_value(state)
        return score

    def _economic_action_score(self, state: GameState, action_id: int) -> float:
        state = self.observable_state(state)
        if not state.economy_enabled:
            return super().action_score(state, action_id)
        action = self.simulator.decode_action(state, action_id)
        if action.kind is ActionKind.MOVE:
            return self._movement_score(state, action)
        if action.kind is ActionKind.EQUIP:
            return self._equipment_score(state, action_id, action)
        options = self.simulator.available_options(state)
        option = options[action.option_index or 0]
        if not option_is_allowed(state, option, self.policy_constraints):
            return -1e9
        if state.pending_node_id and state.floor_map.node(state.pending_node_id).requires_observation and option.operation not in {"leave", "portal_return"}:
            return -1000.0
        if option.operation in {"unknown", "observe", "needs_observation"}:
            return -1000.0
        # The real battle menu can show a recruitment ticket alongside relic
        # and part choices. Finish the pending part reward sequence first, so
        # a corn offered later in this same sequence also receives recruitment
        # growth. This only orders observed actions; it does not redraw loot.
        recruitment_operations = {"select_recruit_ticket", "recruit_reserve", "recruit_temporary", "mechanist_promote", "retain_recruit_ticket", "decline_recruitment"}
        pending_parts = dict(state.event_counters).get(f"{state.pending_node_id}:part_rewards", 0)
        if state.chase_reward_context is not None:
            pending_parts=int(bool(state.chase_reward_context.part_options))
        if option.operation in recruitment_operations and pending_parts > 0 and any(
            candidate.operation == "take" and option_is_allowed(state, candidate, self.policy_constraints)
            for candidate in options
        ):
            return -500.0
        definition = self._definition(option.item_id)
        instance = next((item for item in state.item_instances if item.instance_id == option.instance_id), None)
        category = definition.category if definition is not None else (instance.category if instance else None)
        if category is None and option.item_id and option.item_id.startswith("pool:"):
            category = option.item_id.split(":")[1]
        gold_value = self._gold_value(state)
        if option.operation == "bank_withdraw":
            return self._bank_withdrawal_value(state, options, option)
        if option.operation == "shadow_dance":
            from .events import shadow_dance_outcomes
            stage = min(7, max(0, option.quantity-1))
            if state.resources.hp + option.effect.hp <= 0:
                return -1e6
            expected = shadow_dance_relic_expectation(state.resources.hp, state.resources.max_hp,
                had_reward_on_entry=SHADOW_REWARD_ID in state.inventory, completed_circles=stage)
            outcomes = shadow_dance_outcomes(stage+1, had_relic_on_entry=SHADOW_REWARD_ID in state.inventory)
            other = 0.0
            for outcome in outcomes:
                other += outcome.effect.gold*gold_value + outcome.effect.action_points*self.config.action_point_value + outcome.effect.hope*0.15
                if outcome.item_id and outcome.item_id.startswith("pool:") and outcome.option_id not in {"choice_ro6_normal4_17", "choice_ro6_normal4_24"}:
                    other += self._pool_value(state, outcome.item_id)*outcome.quantity
            # The DP already pays every affordable circle. Subtracting a
            # separate before/after shadow option value would pay that HP twice.
            return expected*self.config.relic_value+other/len(outcomes)+option.effect.hp*0.25
        if option.option_id == "choice_ro6_normal1_1":
            return 80.0
        if option.option_id == "choice_ro6_normal1_2" and any(candidate.option_id == "choice_ro6_normal1_1" for candidate in options):
            return -1e6
        if option.option_id == "choice_ro6_normal3_1":
            return 5000.0
        if option.option_id in LAKE_OFFERING_CHOICES:
            return 40.0 if state.resources.gold > 0 else -1e6
        if option.item_id == "rogue_6_scrap_G_04" and option.operation in {"take", "claim_reward", "event_reward", "claim_scrap"} and any(candidate.operation in {"formal_recruit", "recruit", "recruit_reserve", "recruit_temporary", "select_recruit_ticket"} for candidate in options):
            return 5000.0
        if option.operation == "select_recruit_ticket":
            from .operator_economy import MECHANIST_ID, mechanist_eligible_ticket_ids
            score = 10.0 + 3 * sum(item.item_id == "rogue_6_scrap_G_04" for item in state.item_instances) * gold_value
            if (MECHANIST_ID in state.available_formal_operator_ids and MECHANIST_ID not in state.promoted_operator_ids
                    and option.item_id in mechanist_eligible_ticket_ids()):
                # One displayed ticket is selected; only the ensuing real
                # promotion consumes that ticket and can grant mastery M11.
                score += 100.0
            return score
        if option.operation in {"recruit_reserve", "recruit_temporary"}:
            return 10.0 + 3 * sum(item.item_id == "rogue_6_scrap_G_04" for item in state.item_instances) * gold_value
        if option.operation == 'emergency_hire':
            # The user's observed hire trigger grows each held corn. The
            # temporary employee adds no permanent formal-roster utility.
            gain=3*sum(item.category=='GOODS' and item.item_id=='rogue_6_scrap_G_04'
                for item in state.item_instances)
            return (gain-max(option.price or 0,-option.effect.gold,0))*gold_value
        if category == "UPGRADE_TICKET" and option.operation in {"take", "event_reward", "reward"}:
            mastered = getattr(getattr(self.simulator.economy, "config", None), "mechanist_full_mastery", False)
            if mastered and "char_4230_mcnist" in state.available_formal_operator_ids and "char_4230_mcnist" not in state.promoted_operator_ids:
                return self._new_part_reward_value(state, self._definition("rogue_6_scrap_M_11"))+12.0
            return 1.0
        if option.operation == "retain_recruit_ticket":
            return 0.5
        if option.operation == "decline_recruitment":
            return -1.0
        if action.kind is ActionKind.DISCARD or option.operation == "discard":
            return -(self._held_part_value(state, instance) if instance else self._part_future_value(state, definition)) - (self._sell_quote(state, instance) * gold_value if instance else 0.0)
        if option.operation == "purchase":
            cost = option.price or max(0, -option.effect.gold)
            if category == "RELIC":
                bonus = self._relic_bonus_value(state, definition)
                critical = option.item_id in {"rogue_6_relic_legacy_50", "rogue_6_relic_legacy_93", "rogue_6_relic_legacy_111", "rogue_6_relic_cargo_5", "rogue_6_relic_cargo_6"}
                if state.floor_index < self.config.early_relic_delay_floors and not critical and bonus < cost * gold_value:
                    return -10.0
                score = self.config.relic_value * option.quantity + bonus - cost * gold_value
                reserve = self.config.early_cash_reserve
                if self._remaining_floors(state) > 2 and sum(item.item_id.endswith("P_06") for item in state.item_instances) < self.config.early_p06_reserve_target:
                    reserve = max(reserve, 24)
                if self._remaining_floors(state) > 1 and state.resources.gold - cost < reserve:
                    score -= max(0.0, self.config.relic_value - bonus)
                return score
            if definition is None:
                return -10.0
            shop_node = state.floor_map.node(state.pending_node_id) if state.pending_node_id else None
            economy_config = getattr(self.simulator.economy, "config", None)
            can_sell_here = bool(shop_node and (shop_node.node_type == NodeType.SCRAP_SHOP or
                shop_node.node_type == NodeType.BATTLE_SHOP and getattr(economy_config, "full_tech", False)))
            # A live shop keeps its sale menu during temporary overflow. Three
            # new moss balls can be sold there, leaving exactly the new G07.
            needed_slots = (1 if can_sell_here else 4) if definition.item_id.endswith("G_07") else 1
            roundtrip = self._shop_roundtrip_margin(state, definition, cost)
            overflow_roundtrip = roundtrip is not None and roundtrip > 0 and any(item.item_id.endswith("G_09") for item in state.item_instances)
            if state.resources.parts + needed_slots > state.parts_capacity and not overflow_roundtrip:
                return -100.0
            trade_score = self.config.vine_trade_priority+roundtrip*gold_value if overflow_roundtrip else -1e6
            hypothetical = ItemInstance("quote", definition.item_id, definition.category,
                definition.move_uses or definition.passive_uses, int(definition.sell_price or 0))
            resale = self._sell_quote(state, hypothetical)
            profit = resale - cost
            if category == "GOODS":
                # Same-shop positive-margin trades precede discretionary spend.
                if definition.item_id.endswith("G_07"):
                    shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
                    own_growth = self._sell_quote(state, replace(hypothetical, appraisal=hypothetical.appraisal + 3))
                    moss = self._definition("rogue_6_scrap_G_08")
                    balls = 3 * self._sell_quote(state, ItemInstance("moss_quote", moss.item_id, "GOODS", appraisal=int(moss.sell_price or 0))) if moss and can_sell_here else 0.0
                    bonus = 8 if (can_sell_here and shop and not shop.trade_bonus_paid and
                        getattr(economy_config, "squad", None) == "multilateral_trade") else 0
                    existing_growth = self._acquisition_growth_cash(state, 4)
                    future_growth = self._part_future_value(state, definition) - (definition.sell_price or 0) * gold_value
                    return max(trade_score, (own_growth + balls + bonus + existing_growth - cost) * gold_value + future_growth)
                if profit > 0:
                    return max(trade_score, 45.0 + (profit+self._acquisition_growth_cash(state)) * gold_value)
                return max(trade_score, self._part_future_value(state, definition) + self._acquisition_growth_cash(state)*gold_value - cost * gold_value - 0.5)
            score = self._part_future_value(state, definition) - cost * gold_value
            if category == "MOVE" and self._has_future_navigation(state) and sum(item.uses_remaining or 0 for item in state.item_instances if item.category == "MOVE") < 3:
                score += 12.0
            return max(trade_score, score+self._acquisition_growth_cash(state)*gold_value)
        if option.operation == "sell":
            if category == "RELIC":
                return -100.0
            proceeds = option.effect.gold if option.effect.gold > 0 else option.price
            shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
            trade_bonus = 8 if shop and shop.sold_parts == 2 and not shop.trade_bonus_paid else 0
            proceeds += trade_bonus
            if instance is not None and self._latest_overflow_purchase(state, instance):
                return 10000.0 + proceeds*gold_value
            if category == "GOODS":
                if (definition is not None and instance is not None and definition.item_id.endswith('G_05')
                        and self.config.wave_growth_weight and self._wave_horizon(state) > 0):
                    return self._growth_sale_score(state, definition, instance, proceeds, options)
                if definition is not None and definition.item_id.endswith("G_01") and self._remaining_floors(state) > 1:
                    return proceeds * gold_value
                if definition is not None and instance is not None and definition.item_id.endswith(("G_02", "G_03", "G_04", "G_07", "G_09")) and (self._remaining_floors(state) > 1 or definition.item_id.endswith("G_09")):
                    return self._growth_sale_score(state, definition, instance, proceeds, options)
                return 50.0 + proceeds * gold_value
            future = self._held_part_value(state, instance) if instance else self._part_future_value(state, definition)
            if instance is not None and definition is not None and definition.category == "MOVE":
                all_uses = sum(item.uses_remaining or 0 for item in state.item_instances if item.category == "MOVE")
                free_move = definition.move_ap == 0 or ("rogue_6_relic_cargo_6" in state.inventory and instance.uses_remaining == 1)
                if self._remaining_floors(state) == 1 and not free_move:
                    future *= min(1.0, state.resources.action_points / max(1, all_uses))
            score = proceeds * gold_value - future
            if self.config.sell_for_capacity_upgrade and instance is not None and state.resources.parts >= state.parts_capacity:
                after_sale = replace(state, item_instances=tuple(item for item in state.item_instances if item.instance_id != instance.instance_id),
                    resources=replace(state.resources, gold=state.resources.gold + proceeds, parts=state.resources.parts - 1))
                for candidate in options:
                    target = self._definition(candidate.item_id)
                    if candidate.operation != "purchase" or target is None or not (target.category in {"MOVE", "PASSIVE"} or target.item_id.endswith("G_09")):
                        continue
                    cost = candidate.price or max(0, -candidate.effect.gold)
                    if cost+preparation_cash_target(after_sale, self.policy_constraints) > after_sale.resources.gold:
                        continue
                    utility = self._part_future_value(after_sale, target)
                    if target.category == "MOVE" and self._has_future_navigation(after_sale) and not any(item.category == "MOVE" for item in after_sale.item_instances):
                        utility += 12.0
                    score = max(score, utility - cost * gold_value + proceeds * gold_value - future - 1.0)
            return score
        if option.operation == "refresh":
            cost = option.price or max(0, -option.effect.gold)
            cash_after = state.resources.gold - cost
            pending_kind = state.floor_map.node(state.pending_node_id or "").node_type
            minimum_cash = self.config.green_minimum_refresh_cash if pending_kind == NodeType.SCRAP_SHOP else self.config.minimum_refresh_cash
            if cash_after < minimum_cash:
                return -20.0
            if any(candidate.operation == "purchase" and
                    candidate.is_available(state.resources, state.inventory) and
                    option_is_allowed(state, candidate, self.policy_constraints) and
                    self._definition(candidate.item_id) is not None and
                    self._definition(candidate.item_id).category == "RELIC" and
                    self.action_score(state, self.simulator.ruleset.max_nodes + index) > 0
                    for index, candidate in enumerate(options)):
                return -2.0
            # This is an expected-value preference, not a peek at the next roll.
            if pending_kind == NodeType.SCRAP_SHOP:
                return min(28.0 - cost * gold_value, cash_after * 0.8)
            return min(self.config.relic_value - cost * gold_value, cash_after * 0.3)
        if option.operation == "leave":
            return 0.0
        if option.operation == "starting_reward":
            from .ending_rules import STARTING_REWARD_BY_ID
            spec = STARTING_REWARD_BY_ID[option.option_id]
            return (self.config.relic_value * spec.item_count if spec.item_category == "RELIC" else
                10.0 * spec.item_count if spec.item_category == "MOVE" else 0.0) + spec.gold * gold_value + spec.parts_capacity * 3.0 + self._max_hp_value(state, spec.max_hp)
        if option.operation == "mechanist_promote":
            vehicle = self._definition("rogue_6_scrap_M_11")
            mastered = getattr(getattr(self.simulator.economy, "config", None), "mechanist_full_mastery", False)
            if not mastered or vehicle is None or "char_4230_mcnist" in state.promoted_operator_ids:
                return -1.0
            benefit = self._new_part_reward_value(state, vehicle) + option.effect.hope
            # This free navigation decision uses an actually offered eligible
            # ticket. Resolve a beneficial known upgrade before replanning a
            # route with the new three-use vehicle, including beam navigation.
            return 20000.0 + benefit if benefit > 0 else benefit
        if option.operation == "expedition_source":
            if not self._needs_source_expedition(state):
                return -1.0
            return self.config.relic_value * 9 - (5.0 if option.item_id == "char_4230_mcnist" else 0.0)
        if option.operation == "expedition_inside":
            # A real inside expedition promotes its selected operator on
            # return; full Mechanist mastery then grants one three-use M11.
            mechanist = option.item_id == "char_4230_mcnist" and option.item_id not in state.promoted_operator_ids
            mastered = getattr(getattr(self.simulator.economy, "config", None), "mechanist_full_mastery", False)
            vehicle = self._definition("rogue_6_scrap_M_11") if mechanist and mastered else None
            return 6.0 + (self._part_future_value(state, vehicle) if vehicle else 0.0)
        if option.operation in {"remembrance_parts", "remembrance_key"}:
            benefit = self.config.relic_value * (8 if state.floor <= 4 else 0)
            if option.operation == "remembrance_parts":
                vehicles = [item for item in state.item_instances if item.category == "MOVE"]
                cost = 2 * sum(self._held_part_value(state, item) for item in vehicles) / max(1, len(vehicles))
                return benefit + self.config.relic_value - cost
            return benefit  # A key relic is replaced, so net relic count is zero.
        if option.operation == "portal_enter":
            cost = self._held_part_value(state, instance) if instance else self._part_future_value(state, definition)
            return 40.0 - cost * 0.5
        if option.operation == "portal_return":
            return -10.0 if state.resources.action_points > 0 else 1.0
        if option.operation == "advance":
            exit_node = state.floor_map.node(state.pending_node_id or "")
            bonus = self._exit_vehicle_value(state, exit_node)+self._exit_health_value(state, exit_node)
            if state.resources.action_points <= 2:
                return 30.0+bonus
            if exit_node.node_type == NodeType.EVACUATE:
                return 3.0+bonus
            return 10.0 - max(0, state.resources.action_points - 2) * 4.0+bonus
        if option.operation == "wish_refresh":
            # A refresh does not acquire another item when a free choice remains.
            return -max(1.0, (option.price or -option.effect.gold) * gold_value)
        if option.operation == "cultivate":
            return 10.0 - (self._sell_quote(state, instance) * gold_value if instance else 0.0)
        if option.operation == "exchange":
            if category == "RELIC":
                return -1.0
            return self._exchange_value(state, instance) if instance else -1.0
        if option.operation == "special_battle":
            return 7.0 + option.effect.gold * gold_value
        if option.operation == "expedition":
            return 5.0 if self._remaining_floors(state) > 1 else -1.0
        if option.operation == "parts_capacity":
            return 4.0 if state.resources.parts >= state.parts_capacity - 2 else 1.0
        if option.operation == "supply_voucher":
            return 5.0 if self._remaining_floors(state) > 1 else 2.0
        effect = option.effect
        relics = effect.relics
        if category == "RELIC" and option.operation in {"acquire", "reward", "take", "event_reward", "event_advance"}:
            relics = max(relics, option.quantity)
        score = relics * self.config.relic_value + effect.gold * gold_value
        if category == "RELIC" and definition is not None:
            score += self._relic_bonus_value(state, definition)
        if option.item_id and option.item_id.startswith("pool:") and option.operation in {"acquire", "reward", "take", "event_reward", "event_advance"}:
            # Named pools such as the three duel rewards can contain different
            # categories. Resolve their published pool data, not their label.
            score += option.quantity * self._pool_value(state, option.item_id) - relics * self.config.relic_value
        score += effect.action_points * self.config.action_point_value
        score += effect.parts * (3.0 if state.resources.parts < state.parts_capacity else 0.0)
        score += effect.hope * 0.15 + self._health_effect_value(state, effect)
        if category in {"MOVE", "GOODS", "PASSIVE", "PART"} and option.operation in {"acquire", "reward", "take", "event_reward", "event_advance"} and not (option.item_id or "").startswith("pool:"):
            score += option.quantity * (self._new_part_reward_value(state, definition) if definition else 4.0)
            if definition is not None and definition.category == "MOVE" and self._has_future_navigation(state) and not any(item.category == "MOVE" for item in state.item_instances):
                score += 12.0
        if option.operation in {"event_advance", "lake_resolve", "shadow_dance"}:
            score += 2.0
        if option.battle:
            score += 4.0
        if option.add_items:
            score += 3.0 * len(set(option.add_items) - state.inventory)
        return score

    def _bank_withdrawal_value(self, state: GameState, options, withdrawal) -> float:
        """Withdraw one ingot only toward a concrete valuable purchase deficit."""
        if state.floor == 1 and getattr(self.policy_constraints, "first_floor_full_withdrawal", True):
            return 50000.0
        reserve = preparation_cash_target(state, self.policy_constraints)
        if state.resources.gold < self.policy_constraints.minimum_ingot_reserve:
            return 15.0
        shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
        allowance = max(0, 12-getattr(shop, "entry_withdrawn", 12))
        balance = getattr(state, "bank_balance", 0)
        best = -100.0
        goal = self._known_green_departure_cash_target(state)
        if goal is not None and state.resources.gold < goal:
            deficit = goal-state.resources.gold
            bank_cost = sum(min(getattr(shop, "total_withdrawn", 0)+step, 7)
                for step in range(1, int(deficit)+1))
            # A useful low-cost real sale comes first; an external withdrawal
            # is only valued when its remaining quota/account can close the gap.
            sell_scores = [self.action_score(state, self.simulator.ruleset.max_nodes+index)
                for index, candidate in enumerate(options) if candidate.operation == "sell"
                and candidate.is_available(state.resources, state.inventory)
                and not (candidate.item_id or "").endswith("G_09")
                and not last_redmoss_tool(state, next((item for item in state.item_instances
                    if item.instance_id == candidate.instance_id), None), self.policy_constraints)]
            if max(sell_scores, default=-1.0) > 0:
                return -100.0
            if deficit <= allowance and bank_cost <= balance:
                best = self.config.preparation_cash_shortfall_penalty-0.25*(withdrawal.price or 1)
        for index, candidate in enumerate(options):
            if candidate.operation != "purchase":
                continue
            cost = candidate.price or max(0, -candidate.effect.gold)
            deficit = cost + reserve - state.resources.gold
            if not 0 < deficit <= allowance:
                continue
            bank_cost = sum(min(getattr(shop, "total_withdrawn", 0)+step, 7) for step in range(1, int(deficit)+1))
            if bank_cost > balance:
                continue
            funded = replace(state, resources=replace(state.resources, gold=state.resources.gold+deficit))
            if not option_is_allowed(funded, candidate, self.policy_constraints):
                continue
            value = self.action_score(funded, self.simulator.ruleset.max_nodes+index)
            if value >= 8.0:
                best = max(best, min(18.0, value/max(1, deficit)) - 0.25*(withdrawal.price or 1))
        return best

    def _growth_sale_score(self, state: GameState, definition, instance: ItemInstance, proceeds: float, options) -> float:
        """Retain growing goods unless a concrete use of the proceeds wins.

        Buying for future growth and immediately selling at half price is not
        profitable. A low cash reserve alone is not a reason to destroy growth.
        """

        gold_value = self._gold_value(state)
        future_growth = max(0.0, self._part_future_value(state, definition) - (definition.sell_price or 0) * gold_value)
        if definition.item_id.endswith('G_05'):
            future_growth = max(0.0, self._wave_growth(state, instance.appraisal)*gold_value)
            if future_growth <= 0:
                return 50.0+proceeds*gold_value
        reserve = preparation_cash_target(state, self.policy_constraints)
        if definition.item_id.endswith("G_09"):
            profitable_buys = [candidate for candidate in options if candidate.operation == "purchase"
                and (candidate.price or max(0, -candidate.effect.gold))+reserve <= state.resources.gold
                and (self._shop_roundtrip_margin(state, self._definition(candidate.item_id), candidate.price or max(0, -candidate.effect.gold)) or 0) > 0]
            if profitable_buys:
                return -1000.0
            if future_growth <= 0:
                return 50.0+proceeds*gold_value
        score = -future_growth
        funded = self._project_goods_sale(state, instance, proceeds)
        # Removing a sold instance also removes its menu entries. Locate each
        # public candidate again by stable identity, rather than scoring the
        # action now occupying its old ordinal (often refresh -> leave).
        funded_options = {candidate.option_id: (index, candidate)
            for index, candidate in enumerate(self.simulator.available_options(funded))}
        for original_candidate in options:
            available = funded_options.get(original_candidate.option_id)
            if available is None:
                continue
            index, candidate = available
            cost = candidate.price or max(0, -candidate.effect.gold)
            if not candidate.is_available(funded.resources, funded.inventory) or not option_is_allowed(funded, candidate, self.policy_constraints):
                continue
            if candidate.operation == "refresh":
                if state.resources.gold < cost + 8 <= funded.resources.gold and self.action_score(
                        funded, self.simulator.ruleset.max_nodes + index) > 0:
                    score = max(score, self.config.relic_value - cost * gold_value + proceeds * gold_value - future_growth)
                continue
            target = self._definition(candidate.item_id)
            if candidate.operation != "purchase" or target is None or target.item_id == instance.item_id:
                continue
            needs_capacity = (target.category in {"MOVE", "PASSIVE"} or target.item_id.endswith("G_09")) and state.resources.parts >= state.parts_capacity
            if cost+reserve <= state.resources.gold and not needs_capacity or cost+reserve > state.resources.gold + proceeds:
                continue
            if target.category == "RELIC":
                # The sale is useful only if its actual proceeds make this
                # allowed purchase worth taking, including the live reserves.
                purchase_value = self.action_score(funded, self.simulator.ruleset.max_nodes + index)
            else:
                utility = self._part_future_value(funded, target)
                if target.category == "MOVE" and self._has_future_navigation(funded) and not any(item.category == "MOVE" for item in funded.item_instances):
                    utility += 12.0
                purchase_value = utility - cost * gold_value
            if purchase_value > 0:
                score = max(score, purchase_value + proceeds * gold_value - future_growth)
        return score

    def _exchange_value(self, state: GameState, instance: ItemInstance) -> float:
        definition = self._definition(instance.item_id)
        if definition is None or definition.category == "RELIC" or definition.item_id.endswith("G_09"):
            return -1.0
        if last_redmoss_tool(state, instance, self.policy_constraints):
            return -self.config.redmoss_tool_keep_value
        held = self._held_part_value(state, instance)
        late_candidate = state.floor >= 4 and (instance.item_id.endswith("G_12") or
            instance.item_id.endswith("M_11") and instance.uses_remaining == 1)
        if late_candidate and instance.category == "MOVE":
            free_final_use = definition.move_ap == 0 or "rogue_6_relic_cargo_6" in state.inventory
            if self._remaining_floors(state) == 1 and not free_final_use:
                held *= min(1.0, state.resources.action_points/max(1, definition.move_ap or 0))
            if instance.uses_remaining == 1 and "rogue_6_relic_cargo_5" in state.inventory and (free_final_use or state.resources.action_points > 0 or self._remaining_floors(state)>1):
                held += self._pool_value(state, "pool:MOVE")
        original = max(held, self._sell_quote(state, instance)*self._gold_value(state))
        rank = {"NORMAL": "N", "RARE": "R", "SUPER_RARE": "SR"}.get(definition.rarity)
        if rank is None:
            return -original
        without = replace(state, item_instances=tuple(item for item in state.item_instances if item.instance_id != instance.instance_id),
            resources=replace(state.resources, parts=max(0, state.resources.parts-1)))
        # Replacing one physical SR part does not guarantee a vine. This uses
        # the exact named output pool's conditional frequencies (7/77 G09 in
        # the current observation snapshot), including its other outcomes.
        distribution = self._pool_distribution(without, f"rogue_6:失与得：{rank}零件输出")
        if not distribution:
            return -original
        outcomes = [(self._new_part_reward_value(without, self._definition(item_id)), probability)
            for item_id, probability in distribution.items()]
        expected = sum(value*probability for value, probability in outcomes)
        downside = sum(max(0.0, original-value)*probability for value, probability in outcomes)
        net = expected-original-0.5-self.config.exchange_downside_weight*downside
        # User preference only breaks near-equivalent late swaps. A costly
        # useful final drive or cage event cannot be overridden by this bonus.
        return net+(self.config.late_exchange_preference if late_candidate and net >= 0 else 0.0)

    def _sacrifice_conversion_value(self, state: GameState, node) -> float:
        config = getattr(self.simulator.economy, "config", None)
        counters = dict(state.event_counters)
        if (node.node_id in state.completed or not getattr(config, "full_tech", False) or
                counters.get(node.node_id+":exchange_kind", 0) == 1):
            return 0.0
        remaining = max(0, 2-counters.get(node.node_id+":exchanges", 0))
        values = sorted((self._exchange_value(state, item) for item in state.item_instances
            if item.category in {"MOVE", "GOODS", "PASSIVE"} and
            getattr(self._definition(item.item_id), "can_sacrifice", False)), reverse=True)
        return self.config.sacrifice_conversion_value_weight*sum(max(0.0, value) for value in values[:remaining])

    def _marked_natural_value(self, state: GameState, node) -> float:
        if node.node_id in state.completed or dict(state.event_counters).get("hidden_natural:"+node.node_id, 0) != 1:
            return 0.0
        return self.config.marked_natural_value_weight*self._pool_value(state, "pool:pool_scrap_7")

    def _node_value(self, state: GameState, node) -> float:
        if public_resident_replaces_node(state,node.node_id):
            # Occupation replaces the interaction with combat. Concepts still
            # trigger from the underlying node type, handled independently.
            return self._node_rank_value(state, NodeType.BATTLE_SAVAGE) + self._portal_completion_value(state, node) + self._marked_natural_value(state, node) + (self.config.action_point_value if node.node_id in state.revealed and node.node_type == NodeType.LIGHT and node.node_id not in state.completed else 0.0)
        if node.node_id not in state.revealed and node.node_id not in state.completed:
            distribution = unknown_node_distribution(state, node, self.simulator.ruleset,
                getattr(getattr(self.simulator, "map_generator", None), "config", None))
            return (self.config.unknown_node_value_weight * sum(
                probability*(self._base_node_value(state, replace(node, node_type=kind))+
                    (self._sacrifice_conversion_value(state, node) if kind == NodeType.SACRIFICE else 0.0))
                for kind, probability in distribution.items()) + self._marked_natural_value(state, node)+self._portal_completion_value(state, node))
        return self._known_node_value(state, node)+self._marked_natural_value(state, node)

    def _known_node_value(self, state: GameState, node) -> float:
        return (self._base_node_value(state, node) + self._portal_completion_value(state, node)
            + (self._sacrifice_conversion_value(state, node) if node.node_type == NodeType.SACRIFICE else 0.0))

    def _portal_completion_value(self, state: GameState, node) -> float:
        """Public portal cash bounty, separate from the node interaction."""
        portal = getattr(state, "portal_context", None)
        if (portal is None or node.node_id == state.floor_map.start_node_id or
                node.node_id in state.completed or node.node_id in getattr(portal, "rewarded_node_ids", ())):
            return 0.0
        known = node.node_id in state.revealed
        if portal.variation_id == 7:
            eligible = node.node_type not in {NodeType.EMPTY, NodeType.START} if known else node.is_battle
            return 10.0*self._gold_value(state) if eligible else 0.0
        if portal.variation_id == 9:
            # Purple's public construction family fixes unknown noncombat
            # contents to SACRIFICE; fixed resident battles have coarse battle.
            eligible = node.node_type == NodeType.SACRIFICE if known else not node.is_battle
            return 5.0*self._gold_value(state) if eligible else 0.0
        return 0.0

    def _node_rank_value(self, state: GameState, kind: NodeType) -> float:
        """User's phase-dependent routing order; these are utility, not loot."""
        unit=self.config.relic_value/3.0
        ranks={NodeType.PORTAL:7.5, NodeType.DUEL:7.0,
            NodeType.BATTLE_SAVAGE:5.0,
            NodeType.BATTLE_ELITE:6.0 if state.floor<=3 else 4.0,
            NodeType.INCIDENT:4.0 if state.floor<=3 else 6.0,
            NodeType.BATTLE_NORMAL:3.0 if state.floor<=3 else 2.0,
            NodeType.WISH:2.0 if state.floor<=3 else 3.0}
        return unit*ranks.get(kind,0.0)

    def _transit_arrival_preference(self, state, node, *, equipment=None,
                                   vehicle_available=None, white_dogs=None):
        """Prefer bypassing plain transit while a real vehicle remains usable.

        Painted rewards, public residents/fruit markers and a negative ideal
        source are real development. Required geometric transfers remain
        legal; the route can justify this finite preference by their benefit.
        """
        if (node.node_id not in state.revealed and node.node_id not in state.completed
                or node.node_type not in (NodeType.DOOR,NodeType.EMPTY)):
            return 0.0
        if vehicle_available is None:
            vehicle_available=any(item.category=='MOVE' and (item.uses_remaining or 0)>0
                for item in state.item_instances)
        if not vehicle_available:
            return 0.0
        if white_dogs is None:
            white_dogs=sum(item.item_id.endswith('P_02') for item in state.item_instances)
        if (equipment is not None and white_dogs or
                public_node_has_development(self.simulator,state,node.node_id) or
                self._region_source_priority(state,node,remaining_action_points=state.resources.action_points)>0):
            return 0.0
        return -self.config.relic_value/3.0

    def _base_node_value(self, state: GameState, node) -> float:
        if node.node_id in state.completed and not node.is_exit and node.node_type not in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP}:
            return 0.0
        if node.node_type == NodeType.EXPEDITION and self._needs_source_expedition(state) and getattr(state, "available_formal_operator_ids", ()) and state.portal_context is None:
            available = getattr(getattr(self.simulator, "economy", None), "first_ending_available", None)
            if available and available():
                return self.config.relic_value * 8
        if node.node_type == NodeType.EXPEDITION and state.portal_context is None and "char_4230_mcnist" in state.available_formal_operator_ids and "char_4230_mcnist" not in state.promoted_operator_ids:
            mastered = getattr(getattr(self.simulator.economy, "config", None), "mechanist_full_mastery", False)
            vehicle = self._definition("rogue_6_scrap_M_11") if mastered else None
            if vehicle:
                return 10.0 + self._part_future_value(state, vehicle)
        wish_weight = self.config.early_wish_value_weight if state.floor <= 3 else 1.0
        if node.node_type == NodeType.WISH and self._pool_distribution(state, "rogue_6:得偿所愿"):
            return min(self._node_rank_value(state,NodeType.WISH),self.config.relic_value * wish_weight)
        values = {
            NodeType.BATTLE_NORMAL: self._node_rank_value(state,NodeType.BATTLE_NORMAL),
            NodeType.BATTLE_ELITE: self._node_rank_value(state,NodeType.BATTLE_ELITE),
            NodeType.BATTLE_SAVAGE: self._node_rank_value(state,NodeType.BATTLE_SAVAGE),
            NodeType.DUEL: self._node_rank_value(state,NodeType.DUEL),
            NodeType.BATTLE_BOSS: 12.0,
            NodeType.INCIDENT: self._node_rank_value(state,NodeType.INCIDENT),
            NodeType.WISH: 11.0 * wish_weight,
            NodeType.SACRIFICE: 2.0,
            NodeType.EXPEDITION: 10.0,
            NodeType.PORTAL: self._node_rank_value(state,NodeType.PORTAL),
            NodeType.LIGHT: 3.0,
            NodeType.DOOR: 0.0,
            NodeType.EMPTY: 0.0,
            NodeType.REST: 7.0,
            NodeType.EMPLOY: 1.0,
            NodeType.FINAL: 1.0,
            NodeType.EVACUATE: 1.0,
        }
        if node.node_type == NodeType.PORTAL and not any(item.category == "MOVE" for item in state.item_instances):
            return 0.0
        if node.node_type == NodeType.EMPLOY:
            from .policy_constraints import employ_entry_is_prepared
            if not employ_entry_is_prepared(state):
                return 0.0
        if node.node_type == NodeType.REST:
            return values[NodeType.REST]+self._health_effect_value(state, ResourceDelta(hp=3, max_hp=3))
        if node.node_type in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP}:
            shop = next((shop for shop in state.shops if shop.node_id == node.node_id), None)
            # Regional resale terms are those at the destination shop, not at
            # the current square from which this route is being compared.
            shop_state = replace(state, current_node_id=node.node_id)
            saleable = [item for item in state.item_instances if item.category == "GOODS" or
                (item.category != "RELIC" and self._held_part_value(state, item) < self._sell_quote(shop_state, item) * self._gold_value(state))]
            liquidity = state.resources.gold + sum(self._sell_quote(shop_state, item) for item in saleable)
            entry_value = self._shop_reentry_value(shop_state, shop)
            if shop is not None and shop.visits > 0:
                reserve = self.config.early_cash_reserve if self._remaining_floors(state) > 1 else 0
                relic_prices = [slot.price for slot in shop.stock if not slot.sold and slot.item_id not in state.inventory and slot.item_id not in FORBIDDEN_RELIC_IDS
                    and self._definition(slot.item_id) is not None and self._definition(slot.item_id).category == "RELIC"]
                affordable = [price for price in relic_prices if price <= liquidity - reserve]
                if affordable:
                    budget = max(0.0, liquidity-reserve)
                    count = 0
                    for price in sorted(affordable):
                        if price <= budget:
                            budget -= price
                            count += 1
                    return entry_value + 8.0 + count * self.config.relic_value * self._shop_conversion_weight(state)
                refresh_cost = 4 * (shop.refreshes + 1)
                if shop.refreshes < 4 and liquidity >= refresh_cost + self.config.minimum_refresh_cash:
                    return entry_value + 12.0 + min(15.0, liquidity * 0.2)
                promising_parts = [self._part_future_value(state, self._definition(slot.item_id)) - slot.price * self._gold_value(state)
                    for slot in shop.stock if not slot.sold and slot.price <= liquidity and self._definition(slot.item_id) is not None
                    and self._definition(slot.item_id).category != "RELIC"]
                if max(promising_parts, default=0.0) > 5.0:
                    return entry_value + 8.0 + min(12.0, max(promising_parts) * 0.2)
                return entry_value + 1.0 + len(saleable) * 2.0
            green_bonus = self.config.green_shop_bonus if node.node_type == NodeType.SCRAP_SHOP else 0.0
            conversion = 0.0
            if node.node_type == NodeType.BATTLE_SHOP:
                config = getattr(self.simulator.economy, "config", None)
                # The task profile's explicitly documented synthetic shop
                # layout has seven relic slots. No unseen stock is inspected.
                slots = 7 if config and config.full_tech and config.bank_investment >= 500 else 4
                region = state.region_state
                if region and region.active and region.ideology == "储藏室" and node.node_id in region.covered_node_ids:
                    slots = max(0, slots-3)
                budget = max(0.0, liquidity-(self.config.early_cash_reserve if self._remaining_floors(state)>2 else 0))
                count = min(slots, budget/max(1.0, self.config.unobserved_relic_price_estimate))
                conversion = count * self.config.relic_value * self._shop_conversion_weight(state)
            return entry_value + self.config.shop_value + green_bonus + min(15.0, liquidity * 0.20) + conversion
        return values.get(node.node_type, 5.0)

    def _shop_conversion_weight(self, state: GameState) -> float:
        return self.config.shop_liquidity_weight * (1.0 if self._remaining_floors(state) <= 2 else 0.35)

    def _discovery_value(self, state: GameState, node) -> float:
        newly_visible = set(state.floor_map.adjacency()[node.node_id]) | {node.node_id}
        if node.node_type == NodeType.LIGHT and node.node_id not in state.completed:
            config = getattr(getattr(self.simulator, "economy", None), "config", None)
            radius = 3 if config and config.full_tech else 2
            newly_visible.update(target.node_id for target in state.floor_map.nodes
                if abs(target.row - node.row) + abs(target.col - node.col) <= radius)
        region = getattr(state, "region_state", None)
        if region and region.active and region.ideology in {"弥散虚雾", "“弥散虚雾”"}:
            newly_visible.difference_update(region.covered_node_ids)
        newly_visible.difference_update(state.revealed)
        growth = sum(item.item_id.endswith("G_03") for item in state.item_instances)
        return len(newly_visible) * (self.config.reveal_node_value + growth * self._gold_value(state))

    def _shop_reentry_value(self, state: GameState, shop) -> float:
        config = getattr(getattr(self.simulator, "economy", None), "config", None)
        if config is None or config.squad != "multilateral_trade" or not config.full_tech:
            return 0.0
        moss = ItemInstance("quote", "rogue_6_scrap_G_08", "GOODS", appraisal=2)
        value = self._sell_quote(state, moss)
        for item in state.item_instances:
            growth = 4 if item.item_id.endswith("G_09") else 1 if item.item_id.endswith("G_07") else 0
            if growth:
                value += self._sell_quote(state, replace(item, appraisal=item.appraisal + growth)) - self._sell_quote(state, item)
        if shop is not None and not shop.trade_bonus_paid:
            value += 8.0 if shop.sold_parts == 2 else 8.0 / 3.0
        return value * self._gold_value(state) * self.config.shop_reentry_value_weight

    def _region_source_priority(self, state: GameState, node, *, remaining_action_points: int) -> float:
        """Small user routing preference, not an assumed drop or cash reward.

        Clearing a known negative source matters only while later development
        remains. Beneficial soil and nonremovable regions never earn this bonus.
        """
        region = state.region_state
        if (region is None or not region.active or not region.removable or remaining_action_points <= 0
                or node.node_id != region.source_node_id or node.node_id not in state.revealed
                or node.node_id in state.completed):
            return 0.0
        ideology = region.ideology.strip('“”"')
        policy = (region.policy or '').strip('“”"')
        if ideology in {'希望的沃土', '希望沃土', 'hope_soil'} or policy in {'增益', 'benefit'}:
            return 0.0
        negative_ideology = ideology in {'黑流地脉', '倾斜沙丘', '去温栏', '微型胶囊', '储藏室',
            '易碎同盟', '停止点', '弥散虚雾', 'storage_room'}
        negative_policy = policy in {'改良', '修正', '激进', 'improve', 'correct', 'radical'}
        return 6.0 if negative_ideology or negative_policy else 0.0

    def _movement_score(self, state: GameState, action, *, random_resolution: bool = False) -> float:
        node = state.floor_map.node(action.target_node_id or "")
        cost = action.movement_cost
        develop_fifth = (self.policy_constraints.first_ending_only
            and self.policy_constraints.fifth_floor_exhaust_actions
            and state.floor == 5 and state.portal_context is None)
        score = self._node_value(state, node) - cost * self.config.action_point_value
        plain_ground = (develop_fifth and node.node_id in state.revealed
            and node.node_type == NodeType.EMPTY
            and not public_node_has_development(self.simulator, state, node.node_id))
        if plain_ground:
            score -= self._base_node_value(state, node)
        discovery_value = self._discovery_value(state, node)
        score += discovery_value
        arrival_gold = state.resources.gold
        regional_appraisal_value = 0.0
        region = getattr(state, "region_state", None)
        if region is not None and region.active:
            trajectory = (node.node_id,) if state.equipped_instance_id else action.traversed_node_ids
            affected = sum(region.affects(item) for item in trajectory)
            if region.policy in {"改良", "improve"}:
                score -= min(state.resources.gold, affected * 2) * self._gold_value(state)
                arrival_gold = max(0, arrival_gold-affected*2)
            elif region.policy in {"修正", "correct"}:
                score -= min(state.resources.shield, affected) * 0.1
            elif region.policy in {"激进", "radical"}:
                lost=min(max(0,state.resources.hp-1),affected)
                ap_after=state.resources.action_points-cost
                if node.node_type==NodeType.LIGHT and node.node_id not in state.completed:
                    ap_after+=1
                if any(item.instance_id==state.equipped_instance_id and item.item_id.endswith('M_12')
                        for item in state.item_instances):
                    ap_after+=3
                score += self._travel_health_value(state,node,lost,action_points_after=ap_after)
            elif region.policy in {"增益", "benefit"}:
                regional_appraisal_value = affected * 2 * sum(item.category == "GOODS" for item in state.item_instances) * self._gold_value(state)
                score += regional_appraisal_value
        equipment = next((item for item in state.item_instances if item.instance_id == state.equipped_instance_id), None)
        if equipment is not None:
            definition = self._definition(equipment.item_id)
            if definition is not None and definition.random_move and not random_resolution:
                # The single random-transport action is a UI placeholder, not a
                # controllable destination. Score its observable expectation.
                candidates = list(self.simulator.economy.random_transport_candidates(state))
                if getattr(getattr(self.simulator.economy, "config", None), "random_transport_unknown_first", True):
                    unknown = [candidate for candidate in candidates if candidate.node_id not in state.revealed]
                    candidates = unknown or candidates
                return sum(self._movement_score(state, replace(action, target_node_id=candidate.node_id), random_resolution=True) for candidate in candidates) / max(1, len(candidates))
            if equipment.item_id.endswith("M_10"):
                score += 4 * self._gold_value(state)
                arrival_gold += 4
            if equipment.item_id.endswith("M_12"):
                score += 3 * self.config.action_point_value
            score += self._concept_arrival_value(state, node)
        score += self._transit_arrival_preference(state,node,equipment=equipment)
        arrival_assets=replace(state,item_instances=tuple(replace(item,appraisal=max(0,item.appraisal-2))
            if item.item_id=='rogue_6_scrap_G_10' else item for item in state.item_instances))
        score -= self._green_entry_preparation_penalty(arrival_assets, node, arrival_gold)
        if node.node_id in state.completed and node.node_type not in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP} and not node.is_exit:
            score -= 8.0
        # Known map topology supports planning toward visible destinations.
        adjacency = state.floor_map.adjacency()
        queue = deque([(node.node_id, 0)])
        distances = {node.node_id: 0}
        while queue:
            current, distance = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))
        onward = []
        remaining = state.resources.action_points - cost
        if node.node_type == NodeType.LIGHT and node.node_id not in state.completed:
            remaining += 1
        if equipment is not None and equipment.item_id.endswith("M_12"):
            remaining += 3
        score += self._region_source_priority(state, node, remaining_action_points=remaining)
        available_gear = [(item, self._definition(item.item_id)) for item in state.item_instances
            if item.category == "MOVE" and (item.instance_id != state.equipped_instance_id or (item.uses_remaining or 0) > 1)]
        for target in state.floor_map.nodes:
            distance = distances.get(target.node_id)
            if distance is None or distance == 0 or (target.node_id in state.completed and target.node_type not in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP}):
                continue
            if target.node_id not in state.revealed:
                continue
            if target.node_type in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP, NodeType.BATTLE_ELITE, NodeType.WISH, NodeType.SACRIFICE, NodeType.PORTAL}:
                onward.append(self._node_value(state, target) * 0.55 ** distance)
            # Plan a public, two-move route: walk/transport to this location,
            # then use one remaining vehicle to trigger a known concept node.
            # No destination payload or random transport outcome is previewed.
            concept_value = self._concept_arrival_value(state, target)
            if concept_value > 0 and remaining > 0:
                for _, definition in available_gear:
                    if definition is None or definition.random_move:
                        continue
                    if definition.move_range and (target.row - node.row, target.col - node.col) not in definition.move_range:
                        continue
                    targets = definition.move_target_types
                    if "ALL" not in targets and not ("EVENTS" in targets and not target.is_battle) and target.node_type.value not in targets:
                        continue
                    if int(definition.move_ap or 0) <= remaining:
                        onward.append((self._node_value(state, target) + concept_value) * self.config.concept_route_weight)
                        break
        if not (develop_fifth and remaining <= 0):
            score += max(onward, default=0.0)
        elif plain_ground:
            # No shop or further navigation follows this empty: G03 discovery
            # and soil appraisal remain unsold paper value. Actual concept,
            # resident and marked-item rewards are scored independently.
            score -= discovery_value + regional_appraisal_value
        else:
            # The following chase ends the run: new map information has no
            # later routing use. Keep actual G03 appraisal gained on discovery.
            growth = sum(item.item_id.endswith("G_03") for item in state.item_instances)
            per_reveal = self.config.reveal_node_value + growth * self._gold_value(state)
            if per_reveal:
                score -= discovery_value * self.config.reveal_node_value / per_reveal
        exits = [target for target in state.floor_map.nodes if target.is_exit]
        exit_distance = min((distances.get(target.node_id, 999) for target in exits), default=0)
        # Reserve enough AP to finish. One remaining nonrandom transport can
        # reduce the reserve when its published range reaches an exit directly.
        for item in state.item_instances:
            definition = self._definition(item.item_id)
            if item.category != "MOVE" or definition is None or definition.random_move:
                continue
            if item.instance_id == state.equipped_instance_id and (item.uses_remaining or 0) <= 1:
                continue
            for target in exits:
                if definition.move_range and (target.row - node.row, target.col - node.col) not in definition.move_range:
                    continue
                targets = definition.move_target_types
                if "ALL" not in targets and not ("EVENTS" in targets and not target.is_battle) and target.node_type.value not in targets:
                    continue
                exit_distance = min(exit_distance, int(definition.move_ap or 0))
        exhaust_fifth = develop_fifth
        if exits and not node.is_exit and not exhaust_fifth:
            if remaining < exit_distance:
                score -= self.config.exit_reserve_penalty
            if remaining <= exit_distance + 2:
                score -= exit_distance * self.config.exit_distance_penalty
        if node.is_exit:
            # Both direct completion and the last development step lead to
            # the same final boss. Spending AP itself earns no extra value.
            score += 12.0-self._base_node_value(state, node) if exhaust_fifth else 8.0 - max(0, remaining) * 2.2
            if self._remaining_floors(state) == 1 and not exhaust_fifth:
                score -= min(12.0, state.resources.gold * 0.1)
        elif remaining <= 0:
            score += 12.0 if exhaust_fifth else -5.0
            if plain_ground:
                score -= 0.05  # Prefer direct completion when only transit ties.
        if state.portal_context is None and self._remaining_floors(state) > 1 and (node.is_exit or remaining <= 0):
            unused = sum(max(0, (item.uses_remaining or 0)-int(item.instance_id == state.equipped_instance_id))
                for item in state.item_instances if item.item_id.endswith(("M_04", "M_07")))
            score -= 6.0 * unused
        # One navigation action triggers one wave roll, irrespective of how
        # many revealed empty nodes were traversed. Paper value at the last
        # non-shop action has no use in the physical-relic objective.
        if (self._remaining_floors(state) > 1 or remaining > 0
                or node.node_type in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP}):
            score += sum(self._wave_growth(state, item.appraisal, moves=1)
                for item in state.item_instances if item.item_id.endswith('G_05'))*self._gold_value(state)
        return score

    def _concept_arrival_value(self, state: GameState, node) -> float:
        value = 0.0
        fresh = node.node_id not in state.completed
        distribution = ({node.node_type: 1.0} if node.node_id in state.revealed else
            unknown_node_distribution(state, node, self.simulator.ruleset,
                getattr(getattr(self.simulator, "map_generator", None), "config", None))
            if fresh and any(item.item_id.endswith(("P_05", "P_06")) for item in state.item_instances) else {})
        for item in state.item_instances:
            if item.item_id.endswith("P_01") and node.is_battle:
                value += self.config.relic_value
            elif item.item_id.endswith("P_02") and not node.is_battle:
                value += self.config.relic_value
            elif item.item_id.endswith("P_05") and fresh:
                value += self.config.relic_value * distribution.get(NodeType.WISH, 0.0)
            elif item.item_id.endswith("P_06") and fresh:
                value += 2 * self.config.relic_value * distribution.get(NodeType.SACRIFICE, 0.0)
        return value

    def _equipment_score(self, state: GameState, action_id: int, action) -> float:
        if action.equipment_instance_id == state.equipped_instance_id:
            return -100.0
        baseline = [self._movement_score(state, candidate) for candidate in self.simulator.legal_actions(state)
                    if candidate.kind is ActionKind.MOVE]
        preview = self.simulator.transition(state, action_id).next_state
        alternatives = [self._movement_score(preview, candidate) for candidate in self.simulator.legal_actions(preview)
                        if candidate.kind is ActionKind.MOVE]
        if not alternatives:
            return -100.0
        best = max(alternatives)
        if best <= max(baseline, default=-100.0) + self.config.equipment_switch_margin:
            return -50.0
        return best - 0.25


def choose_random_action(
    simulator: BlackflowSimulator,
    state: GameState,
    rng: random.Random,
) -> int:
    legal = allowed_action_ids(simulator, state)
    if not legal:
        raise RuntimeError("non-terminal state has no legal actions")
    return rng.choice(legal)


def choose_heuristic_action(
    evaluator: HeuristicEvaluator,
    state: GameState,
) -> int:
    legal = allowed_action_ids(evaluator.simulator, state, evaluator.policy_constraints)
    if not legal:
        raise RuntimeError("non-terminal state has no legal actions")
    return max(legal, key=lambda action_id: (evaluator.action_score(state, action_id), -action_id))
