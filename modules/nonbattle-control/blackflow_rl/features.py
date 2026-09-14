from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .domain import GameState, NodeType
from .simulator import BlackflowSimulator
from .policy_constraints import (DEFAULT_POLICY_CONSTRAINTS, allowed_action_ids,
    redmoss_preparation_active, preparation_cash_target, policy_mask_schema)
from .health_planning import SHADOW_EVENT_NAME, SHADOW_REWARD_ID, shadow_dance_relic_expectation


OBSERVED_NODE_LABELS = tuple(item.value for item in NodeType) + (
    "UNKNOWN_MYSTERY",
    "UNKNOWN_FEROCITY",
)
_NODE_LABEL_INDEX = {label: index for index, label in enumerate(OBSERVED_NODE_LABELS)}

RESOURCE_FIELDS = (
    "hp",
    "max_hp",
    "shield",
    "gold",
    "hope",
    "parts",
    "relics",
    "tickets",
    "team_strength",
    "action_points",
)
RESOURCE_SCALES = np.asarray((8, 8, 10, 20, 10, 5, 5, 5, 10, 5), dtype=np.float32)
INVENTORY_FLAGS = ("alpha", "beta", "beacon", "cage", "ending_3_key", "processed")
ITEM_CATEGORIES = ("RELIC", "MOVE", "GOODS", "PASSIVE")
OPTION_OPERATIONS = (
    "event", "purchase", "sell", "refresh", "equip", "discard", "leave",
    "event_reward", "take", "wish_refresh", "exchange", "advance", "cultivate", "unknown",
    "portal_enter", "portal_return", "expedition", "special_battle", "event_advance",
    "starting_reward", "expedition_inside", "expedition_source", "mechanist_promote", "bank_withdraw", "recruit_reserve", "claim_scrap",
    "retain_recruit_ticket", "decline_recruitment", "select_recruit_ticket", "recruit_temporary", "emergency_hire", "employment_refresh",
    "remembrance_parts", "remembrance_key", "fate_mark", "fate_leave", "fate_commit", "fate_battle",
    "cave_draw", "event_mark", "event_move", "lake_resolve", "needs_observation",
    "parts_capacity", "red_moss", "shadow_dance", "squad_capacity", "supply_voucher",
    "broker_reserve", "broker_random_six",
)
REGION_IDEOLOGIES = ("黑流地脉", "倾斜沙丘", "去温栏", "微型胶囊", "储藏室", "易碎同盟", "停止点", "弥散虚雾", "希望的沃土")
REGION_POLICIES = ("改良", "修正", "激进", "增益")
CHOICE_ID_BITS = 16
STAGE_ID_BYTES = 32
GOODS_SUMMARY_FIELDS = ("sum", "minimum", "maximum", "sum_clipped_at_two")
SHOP_SCALAR_FIELDS = ("observed", "refreshes", "remaining_refreshes", "visits",
    "sold_parts", "trade_bonus_paid", "sold_slots", "lost_slots", "total_withdrawn", "entry_withdrawn")
SHOP_CATEGORIES = ITEM_CATEGORIES + ("RECRUIT_TICKET", "UPGRADE_TICKET", "SERVICE_TICKET", "SERVICE_TOOL")


@lru_cache(maxsize=1)
def _public_client_choice_ids():
    """Vocabulary labels only; never load hidden scenes or future outcomes."""
    path = Path(__file__).resolve().parents[1] / "data/evidence/rogue6_client_choice_snapshot_v1.json"
    values = json.loads(path.read_text(encoding="utf-8"))["choices"]
    return tuple(sorted({choice["id"] for choice in values}))


@dataclass(frozen=True, slots=True)
class EncodedState:
    node_features: np.ndarray
    adjacency: np.ndarray
    node_mask: np.ndarray
    global_features: np.ndarray
    option_features: np.ndarray
    option_observation_mask: np.ndarray
    action_mask: np.ndarray

    def as_torch(self, device: str | None = None) -> dict[str, Any]:
        """Convert one sample without importing torch at package import time."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on minimal installs
            raise RuntimeError(
                "PyTorch is required for neural inference; install the training dependencies"
            ) from exc
        target = torch.device(device) if device is not None else None
        return {
            "node_features": torch.as_tensor(self.node_features, device=target),
            "adjacency": torch.as_tensor(self.adjacency, device=target),
            "node_mask": torch.as_tensor(self.node_mask, device=target),
            "global_features": torch.as_tensor(self.global_features, device=target),
            "option_features": torch.as_tensor(self.option_features, device=target),
            "option_observation_mask": torch.as_tensor(self.option_observation_mask, device=target),
            "action_mask": torch.as_tensor(self.action_mask, device=target),
        }


class FeatureEncoder:
    """Permutation-equivariant padded graph and candidate-action encoder."""

    NODE_SCALAR_DIM = 17
    GLOBAL_BASE_DIM = 14
    ECONOMY_GLOBAL_DIM = 54 + len(REGION_IDEOLOGIES) + len(REGION_POLICIES)
    OPTION_EXTRA_DIM = 4 + len(ITEM_CATEGORIES) + len(OPTION_OPERATIONS) + 12 + CHOICE_ID_BITS + STAGE_ID_BYTES
    SCHEMA_VERSION = 6

    def __init__(
        self,
        simulator: BlackflowSimulator,
        *,
        constraints=DEFAULT_POLICY_CONSTRAINTS,
        full_observability: bool = False,
    ) -> None:
        self.simulator = simulator
        self.ruleset = simulator.ruleset
        self.policy_constraints = constraints
        self.full_observability = full_observability
        catalog = getattr(getattr(simulator, "economy", None), "catalog", None)
        self.item_identities = tuple(sorted({definition.canonical_id
            for definition in catalog.items.values() if definition.category in ITEM_CATEGORIES})) if catalog else ()
        self._item_identity_index = {item_id: index for index, item_id in enumerate(self.item_identities)}
        self.move_identities = tuple(sorted({item.canonical_id for item in catalog.items.values()
            if item.category == "MOVE"})) if catalog else ()
        self.goods_identities = tuple(sorted({item.canonical_id for item in catalog.items.values()
            if item.category == "GOODS"})) if catalog else ()
        self.move_use_bins = tuple(range(1 + max((item.move_uses or 0 for item in catalog.items.values()
            if item.category == "MOVE"), default=0))) if catalog else (0,)
        self.client_choice_ids = _public_client_choice_ids()
        if len(self.client_choice_ids) >= 2**CHOICE_ID_BITS:
            raise ValueError("public choice vocabulary exceeds its exact binary encoding")
        self._client_choice_index = {choice_id: index+1 for index, choice_id in enumerate(self.client_choice_ids)}

    @property
    def inventory_detail_dim(self) -> int:
        # Exact remaining-use histogram by type, equipped uses, expiry count;
        # compact appraisal distribution summaries by natural-object type.
        return len(self.move_identities)*(len(self.move_use_bins)+2) + len(self.goods_identities)*len(GOODS_SUMMARY_FIELDS)

    @property
    def shop_node_dim(self) -> int:
        return len(SHOP_SCALAR_FIELDS) + 2*len(SHOP_CATEGORIES) + 2*len(self.item_identities)

    @property
    def node_feature_dim(self) -> int:
        return len(OBSERVED_NODE_LABELS) + self.NODE_SCALAR_DIM + self.shop_node_dim

    @property
    def global_feature_dim(self) -> int:
        return self.GLOBAL_BASE_DIM + len(INVENTORY_FLAGS) + self.inventory_detail_dim + self.ECONOMY_GLOBAL_DIM + len(self.item_identities)

    @property
    def option_feature_dim(self) -> int:
        return len(RESOURCE_FIELDS) + self.OPTION_EXTRA_DIM + len(self.item_identities)

    @property
    def schema(self) -> dict[str, Any]:
        """Identity of action/observation semantics, independently of map rules."""

        return {
            "version": self.SCHEMA_VERSION,
            "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "health_feature_implementation_sha256": sha256(Path(__file__).with_name("health_planning.py").read_bytes()).hexdigest(),
            "full_observability": self.full_observability,
            "empty_node_visibility": "entered-region EMPTY is intrinsically public; never UNKNOWN_MYSTERY; pre-entry and future maps remain masked",
            "node_feature_dim": self.node_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "option_feature_dim": self.option_feature_dim,
            "max_nodes": self.ruleset.max_nodes,
            "max_options": self.ruleset.max_options,
            "action_size": self.simulator.action_size,
            "item_identities": self.item_identities,
            "move_identities": self.move_identities,
            "move_use_bins": self.move_use_bins,
            "goods_identities": self.goods_identities,
            "goods_summary_fields": GOODS_SUMMARY_FIELDS,
            "inventory_detail_order": "per MOVE type: use histogram / 20, equipped uses / 10, expiring count / 20; then per GOODS type: sum/min/max / 50, sum clipped at two / 20",
            "shop_node_features": "observed visits only; shop scalars, category counts / 10, minimum current category prices / 30, item counts / 10, minimum current item prices / 30",
            "shop_scalar_fields": SHOP_SCALAR_FIELDS,
            "shop_categories": SHOP_CATEGORIES,
            "client_choice_ids": self.client_choice_ids,
            "choice_id_encoding": f"one-based sorted public client vocabulary index as {CHOICE_ID_BITS} binary digits; all-zero for non-client or unknown choice IDs",
            "stage_id_encoding": f"only displayed special_battle/fate_battle item_id ASCII bytes / 127, padded to {STAGE_ID_BYTES}; no node.stage_id input",
            "terminal_progress_fields": "relic_layer:rogue_6_relic_artifact_1 / 2 and its paid:rogue_6_relic_artifact_2 count",
            "policy_mask": policy_mask_schema(self.policy_constraints),
            "option_observation_mask": "all displayed options, including currently unaffordable choices; padding false",
            "advisory_features": {
                "redmoss_preparation_active": "configured preparation flag; zero in task_scope_only mode",
                "preparation_cash_target": "configured cash target / 20; zero in task_scope_only mode",
            },
        }

    @property
    def schema_sha256(self) -> str:
        return sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()

    def encode(self, state: GameState) -> EncodedState:
        if not self.full_observability:
            # Also hides the first map while a pre-exploration reward is being
            # chosen; geometry and battle-category bits must not leak there.
            state = self.simulator.belief_state(state)
        max_nodes = self.ruleset.max_nodes
        max_options = self.ruleset.max_options
        floor_map = state.floor_map
        node_features = np.zeros(
            (max_nodes, self.node_feature_dim), dtype=np.float32
        )
        adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        node_mask = np.zeros(max_nodes, dtype=np.bool_)
        option_features = np.zeros(
            (max_options, self.option_feature_dim), dtype=np.float32
        )
        action_mask = np.zeros(max_nodes + max_options, dtype=np.bool_)
        option_observation_mask = np.zeros(max_options, dtype=np.bool_)

        frontier = self.simulator.reachable_frontier(state) if not state.terminal else {}
        region = getattr(state, "region_state", None)
        active_region = region is not None and region.active
        residents = getattr(state, "resident_context", None)
        occupied = residents.occupied_node_ids if residents is not None else frozenset()
        public_counters = dict(state.event_counters)
        paths = {action.target_node_id: action.traversed_node_ids for action in self.simulator.legal_actions(state)
            if action.target_node_id is not None}
        base_ap = self.ruleset.floor(state.floor).action_points
        for node in floor_map.nodes:
            node_mask[node.index] = True
            observed = self.full_observability or node.node_id in state.revealed
            if observed:
                type_label = node.node_type.value
            else:
                type_label = (
                    "UNKNOWN_FEROCITY" if node.is_battle else "UNKNOWN_MYSTERY"
                )
            node_features[node.index, _NODE_LABEL_INDEX[type_label]] = 1.0
            offset = len(OBSERVED_NODE_LABELS)
            movement_cost = frontier.get(node.node_id, 0)
            node_features[node.index, offset:offset+self.NODE_SCALAR_DIM] = np.asarray(
                (
                    node.node_id == state.current_node_id,
                    node.node_id in state.completed,
                    observed,
                    node.is_exit and observed,
                    node.node_id == state.pending_node_id,
                    node.node_id in frontier,
                    node.row / max(1, floor_map.height - 1),
                    node.col / max(1, floor_map.width - 1),
                    node.distance_from_start / 15.0,
                    movement_cost / max(1, base_ap),
                    bool(node.options) and node.node_id == state.pending_node_id,
                    node.is_battle,
                    bool(active_region and region.affects(node.node_id)),
                    bool(active_region and region.source_node_id == node.node_id),
                    sum(region.affects(item) for item in paths.get(node.node_id, ())) / 5.0 if active_region else 0.0,
                    node.node_id in occupied,
                    public_counters.get("hidden_natural:"+node.node_id, 0) == 1,
                ),
                dtype=np.float32,
            )
            node_features[node.index, offset+self.NODE_SCALAR_DIM:] = self._public_shop_features(state, node)

        id_to_index = {node.node_id: node.index for node in floor_map.nodes}
        for left, right in floor_map.edges:
            left_index, right_index = id_to_index[left], id_to_index[right]
            adjacency[left_index, right_index] = 1.0
            adjacency[right_index, left_index] = 1.0

        legal_ids = allowed_action_ids(self.simulator, state, self.policy_constraints)
        for action_id in legal_ids:
            action_mask[action_id] = True

        available_options = getattr(self.simulator, "available_options", None)
        if available_options is not None:
            options = available_options(state)
        elif state.pending_node_id is not None:
            options = floor_map.node(state.pending_node_id).options
        else:
            options = ()
        if options:
            for index, option in enumerate(options[:max_options]):
                option_observation_mask[index] = True
                delta = np.asarray(
                    [getattr(option.effect, name) for name in RESOURCE_FIELDS],
                    dtype=np.float32,
                ) / RESOURCE_SCALES
                option_features[index] = np.concatenate(
                    (
                        delta,
                        np.asarray(
                            (
                                option.battle,
                                len(option.add_items) / 3.0,
                                len(option.remove_items) / 3.0,
                                option.is_available(state.resources, state.inventory),
                            ),
                            dtype=np.float32,
                        ),
                        self._economic_option_features(state, option),
                    )
                )

        resources = state.resources
        completed_here = sum(
            node.node_id in state.completed for node in floor_map.nodes
        )
        global_base = np.asarray(
            (
                (state.floor - 1) / max(1, len(state.maps) - 1),
                completed_here / max(1, len(floor_map.nodes)),
                resources.action_points / max(1, base_ap),
                resources.hp / max(1, resources.max_hp),
                resources.max_hp / 20.0,
                resources.shield / 20.0,
                resources.gold / 50.0,
                resources.hope / 30.0,
                resources.parts / 10.0,
                resources.relics / 10.0,
                resources.tickets / 10.0,
                resources.team_strength / 20.0,
                state.pending_node_id is not None,
                state.chase_count / max(1, len(state.maps)),
            ),
            dtype=np.float32,
        )
        inventory = np.asarray(
            [flag in state.inventory for flag in INVENTORY_FLAGS], dtype=np.float32
        )
        instances = state.item_instances
        category_counts = [sum(item.category == category for item in instances) / 20.0 for category in ITEM_CATEGORIES]
        current_shop = next((shop for shop in state.shops if shop.node_id == state.pending_node_id), None)
        portal = getattr(state, "portal_context", None)
        counters = dict(state.event_counters)
        shadow_pending = bool(state.pending_node_id and state.floor_map.node(state.pending_node_id).event_name == SHADOW_EVENT_NAME)
        shadow_stage = min(7, counters.get(str(state.pending_node_id)+":stage", 0)) if shadow_pending else 0
        economy_global = np.asarray(
            (
                counters.get("relic_layer:rogue_6_relic_artifact_1", 0) / 2.0,
                counters.get("relic_layer:rogue_6_relic_artifact_1:paid:rogue_6_relic_artifact_2", 0),
                *category_counts,
                state.economy_enabled,
                state.parts_capacity / 20.0,
                (state.parts_capacity - resources.parts) / 20.0,
                state.supply_vouchers / 10.0,
                state.equipped_instance_id is not None,
                sum((item.uses_remaining or 0) for item in instances if item.category == "MOVE") / 30.0,
                sum(item.appraisal for item in instances if item.category == "GOODS") / 50.0,
                current_shop.refreshes / 10.0 if current_shop else 0.0,
                current_shop.sold_parts / 10.0 if current_shop else 0.0,
                current_shop.trade_bonus_paid if current_shop else False,
                portal is not None,
                state.resources.action_points / 10.0 if portal else 0.0,
                portal.outer_action_points / 10.0 if portal else 0.0,
                portal.variation_id / 10.0 if portal else 0.0,
                counters.get("commander_level", 1) / 10.0,
                counters.get("commander_exp", 0) / 100.0,
                active_region,
                region.effect_tier / 3.0 if active_region else 0.0,
                *(bool(active_region and region.ideology == name) for name in REGION_IDEOLOGIES),
                *(bool(active_region and region.policy == name) for name in REGION_POLICIES),
                len(occupied) / 6.0,
                state.current_node_id in occupied,
                len(state.formal_operator_ids) / 6.0,
                len(state.available_formal_operator_ids) / 6.0,
                len(state.promoted_operator_ids) / 6.0,
                any(request.kind == "source" for request in state.expeditions),
                bool(counters.get("starting_reward_pending", 0)),
                "char_4230_mcnist" in state.available_formal_operator_ids,
                "char_4230_mcnist" in state.promoted_operator_ids,
                getattr(state, "bank_balance", 0) / 1000.0,
                getattr(state, "total_bank_withdrawn", 0) / 100.0,
                getattr(state, "bank_balance_spent", 0) / 1000.0,
                getattr(current_shop, "entry_withdrawn", 0) / 12.0 if current_shop else 0.0,
                len(state.pending_recruit_ticket_ids) / 3.0,
                len(state.stored_recruit_ticket_ids) / 3.0,
                (int(bool(state.chase_reward_context.part_options)) if state.chase_reward_context
                 else counters.get(f"{state.pending_node_id}:part_rewards", 0)) / 3.0,
                len(getattr(state, "unknown_recruit_opportunities", ())) / 3.0,
                getattr(state, "chase_reward_pending", False),
                len(getattr(state, "pending_recruit_candidate_groups", ())) / 3.0,
                sum(len(group.candidate_item_ids) for group in getattr(state, "pending_recruit_candidate_groups", ())) / 9.0,
                (redmoss_preparation_active(state, self.policy_constraints)
                    if self.policy_constraints.strategy_advice_enabled else False),
                sum(item.item_id.endswith(("G_01", "G_12")) for item in instances) / 2.0,
                SHADOW_EVENT_NAME in state.seen_event_names,
                shadow_stage / 8.0 if shadow_pending else -1.0,
                shadow_dance_relic_expectation(resources.hp, resources.max_hp, had_reward_on_entry=SHADOW_REWARD_ID in state.inventory,
                    completed_circles=shadow_stage) / 5.0 if SHADOW_EVENT_NAME not in state.seen_event_names else 0.0,
                (preparation_cash_target(state, self.policy_constraints) / 20.0
                    if self.policy_constraints.strategy_advice_enabled else 0.0),
                len(getattr(state, "temporary_recruit_offers", ())) / 3.0,
                len(state.employment_context.emergency_operators) / 6.0 if getattr(state, "employment_context", None) else 0.0,
                len(state.chase_reward_context.relic_options) / 3.0 if state.chase_reward_context else 0.0,
                len(state.chase_reward_context.part_options) / 3.0 if state.chase_reward_context else 0.0,
            ),
            dtype=np.float32,
        )
        item_inventory = np.zeros(len(self.item_identities), dtype=np.float32)
        for item in state.item_instances:
            item_inventory += self._item_identity_features(item.item_id)
        global_features = np.concatenate((global_base, inventory, self._inventory_details(state), economy_global, item_inventory))

        return EncodedState(
            node_features=node_features,
            adjacency=adjacency,
            node_mask=node_mask,
            global_features=global_features,
            option_features=option_features,
            option_observation_mask=option_observation_mask,
            action_mask=action_mask,
        )

    def _economic_option_features(self, state: GameState, option: Any) -> np.ndarray:
        from .operator_economy import mechanist_eligible_ticket_ids
        economy = getattr(self.simulator, "economy", None)
        catalog = getattr(economy, "catalog", None)
        definition = catalog.items.get(option.item_id) if catalog is not None and option.item_id else None
        instance = next((item for item in state.item_instances if item.instance_id == option.instance_id), None)
        category = definition.category if definition is not None else (instance.category if instance else None)
        starting = None
        if option.operation == "starting_reward":
            from .ending_rules import STARTING_REWARD_BY_ID
            starting = STARTING_REWARD_BY_ID[option.option_id]
            category = starting.item_category
        if category is None and option.item_id and option.item_id.startswith("pool:"):
            category = option.item_id.split(":")[1]
        category_features = [float(category == name) for name in ITEM_CATEGORIES]
        if catalog is not None and option.item_id and option.item_id.startswith("pool:"):
            pool_id = option.item_id.removeprefix("pool:")
            if pool_id in catalog.observed_pools:
                weights = [(catalog.items.get(item_id), weight) for item_id, weight in catalog.observed_weights(pool_id)]
                weights = [(item, weight) for item, weight in weights if item is not None]
                total = sum(weight for _, weight in weights)
                if total:
                    category_features = [sum(weight for item, weight in weights if item.category == name) / total for name in ITEM_CATEGORIES]
        scalars = np.asarray(
            (
                *category_features,
                *(option.operation == name for name in OPTION_OPERATIONS),
                option.price / 30.0,
                (starting.item_count if starting else option.quantity) / 5.0,
                ({"NORMAL": 1, "RARE": 2, "SUPER_RARE": 3, "SPECIAL": 4}.get(definition.rarity, 0) / 4.0 if definition is not None else
                    {"普通": 0.25, "稀有": 0.5}.get(starting.client_rarity_label, 0.0) if starting else 0.0),
                (definition.base_buy_price or 0) / 30.0 if definition is not None else 0.0,
                (definition.sell_price or 0) / 30.0 if definition is not None else 0.0,
                (instance.uses_remaining or 0) / 10.0 if instance else ((definition.move_uses or 0) / 10.0 if definition is not None else 0.0),
                instance.appraisal / 30.0 if instance else 0.0,
                option.ends_node,
                (starting.parts_capacity if starting else int(option.operation == "parts_capacity")) / 5.0,
                option.item_id == "char_4230_mcnist",
                option.operation == "select_recruit_ticket" and option.item_id in mechanist_eligible_ticket_ids(),
                category == "UPGRADE_TICKET",
            ),
            dtype=np.float32,
        )
        identity = self._item_identity_features(option.item_id)
        if catalog is not None and option.item_id and option.item_id.startswith("pool:"):
            pool_id = option.item_id.removeprefix("pool:")
            if pool_id in catalog.observed_pools:
                weights = catalog.observed_weights(pool_id)
                total = sum(weight for _, weight in weights)
                if total:
                    for item_id, weight in weights:
                        identity += self._item_identity_features(item_id) * weight / total
        # Keep the final twelve physical option scalars adjacent to identity;
        # the semantic label is extra input, never a substituted value score.
        return np.concatenate((scalars[:-12], self._public_option_semantics(option), scalars[-12:], identity))

    def _public_option_semantics(self, option: Any) -> np.ndarray:
        """Distinguish visible branches without indexing private instance IDs."""
        number = self._client_choice_index.get(option.option_id, 0)
        choice = np.asarray([(number >> bit) & 1 for bit in range(CHOICE_ID_BITS)], dtype=np.float32)
        stage = np.zeros(STAGE_ID_BYTES, dtype=np.float32)
        if option.operation in {"special_battle", "fate_battle"} and option.item_id:
            raw = option.item_id.encode("ascii")
            if len(raw) > STAGE_ID_BYTES:
                raise ValueError("displayed stage identifier exceeds the feature schema's exact byte encoding")
            stage[:len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)/127.0
        return np.concatenate((choice, stage))

    def _inventory_details(self, state: GameState) -> np.ndarray:
        catalog = getattr(getattr(self.simulator, "economy", None), "catalog", None)
        result = []
        for item_id in self.move_identities:
            held = tuple(item for item in state.item_instances if item.item_id == item_id and item.category == "MOVE")
            result.extend(sum(item.uses_remaining == uses for item in held)/20.0 for uses in self.move_use_bins)
            result.append(sum((item.uses_remaining or 0) for item in held
                if item.instance_id == state.equipped_instance_id)/10.0)
            result.append(len(held)*bool(catalog.items[item_id].expires_on_floor_change)/20.0)
        for item_id in self.goods_identities:
            values = tuple(item.appraisal for item in state.item_instances
                if item.item_id == item_id and item.category == "GOODS")
            result.extend((sum(values)/50.0, min(values, default=0)/50.0,
                max(values, default=0)/50.0, sum(min(max(0, value), 2) for value in values)/20.0))
        return np.asarray(result, dtype=np.float32)

    def _public_shop_features(self, state: GameState, node: Any) -> np.ndarray:
        """Remember a genuinely visited shelf; never inspect generated stock."""
        result = np.zeros(self.shop_node_dim, dtype=np.float32)
        shop = next((shop for shop in state.shops if shop.node_id == node.node_id
            and shop.visits > 0 and (node.node_id in state.completed or node.node_id == state.pending_node_id)), None)
        if shop is None or node.node_type not in {NodeType.SCRAP_SHOP, NodeType.BATTLE_SHOP}:
            return result
        engine = self.simulator.economy
        scalar_count = len(SHOP_SCALAR_FIELDS)
        result[:scalar_count] = (1, shop.refreshes/4.0,
            max(0, 4-shop.refreshes)/4.0 if engine.config.full_tech else 0,
            shop.visits/10.0, shop.sold_parts/10.0, shop.trade_bonus_paid,
            sum(slot.sold for slot in shop.stock)/10.0, shop.lost_slots/10.0,
            shop.total_withdrawn/20.0, shop.entry_withdrawn/12.0)
        counts = np.zeros(len(self.item_identities), dtype=np.float32)
        prices = np.zeros(len(self.item_identities), dtype=np.float32)
        category_counts = np.zeros(len(SHOP_CATEGORIES), dtype=np.float32)
        category_prices = np.zeros(len(SHOP_CATEGORIES), dtype=np.float32)
        projected = replace(state, current_node_id=node.node_id)
        for option in engine.shop_options(projected, node):
            if option.operation != "purchase":
                continue
            definition = engine.catalog.items.get(option.item_id)
            category_name = definition.category if definition else {
                "service:ticket": "SERVICE_TICKET", "service:tool": "SERVICE_TOOL"}.get(option.item_id)
            if category_name not in SHOP_CATEGORIES:
                continue
            index = self._item_identity_index.get(definition.canonical_id) if definition else None
            category = SHOP_CATEGORIES.index(category_name)
            if index is not None:
                if counts[index] == 0 or option.price < prices[index]:
                    prices[index] = option.price
                counts[index] += 1
            if category_counts[category] == 0 or option.price < category_prices[category]:
                category_prices[category] = option.price
            category_counts[category] += 1
        result[scalar_count:] = np.concatenate((category_counts/10.0, category_prices/30.0,
            counts/10.0, prices/30.0))
        return result

    def _item_identity_features(self, item_id: str | None) -> np.ndarray:
        """Public identities keep same-price relics and concept combinations distinct."""

        result = np.zeros(len(self.item_identities), dtype=np.float32)
        catalog = getattr(getattr(self.simulator, "economy", None), "catalog", None)
        definition = catalog.items.get(item_id) if catalog is not None and item_id else None
        index = self._item_identity_index.get(definition.canonical_id) if definition is not None else None
        if index is not None:
            result[index] = 1.0
        return result


def stack_encoded(samples: list[EncodedState]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot stack an empty encoded batch")
    return {
        field: np.stack([getattr(sample, field) for sample in samples], axis=0)
        for field in EncodedState.__dataclass_fields__
    }
