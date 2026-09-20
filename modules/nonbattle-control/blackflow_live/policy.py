"""Inference-only adapter from real observations to the selected neural weights.

No simulator, episode, random reward, teacher, transition or synthetic resource
state is constructed here. Unknown screenshot information stays absent from
the observation; zero-filled unused input positions are an explicit partial
observation encoding, not predictions of the game's hidden state.
"""
from __future__ import annotations

from hashlib import sha256
from dataclasses import replace
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from .models import LiveObservation, ObservedAction, PolicyDecision


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROUTE_RESOURCES = ("action_points", "hp", "max_hp", "gold", "hope", "parts", "relics")
BLOCKED_SCENES = frozenset({"battle", "battle_start", "battle_prepare", "squad", "combat", "loading", "unknown", "ending", "ending_complete"})
BLOCKED_OPERATIONS = frozenset({"battle", "battle_start", "start_battle", "special_battle", "fate_battle", "combat", "surrender", "restart", "abandon_run", "expedition_source"})
BLOCKED_LABELS = ("开始战斗", "开始作战", "进入战斗", "作战开始", "开始行动", "放弃探索", "放弃本次", "重新开始", "start battle", "start operation")
OTHER_ENDING_LABELS = ("二结局", "三结局", "四结局", "结局二", "结局三", "结局四", "第二结局", "第三结局", "第四结局")
RECRUIT_OPERATIONS = frozenset({"recruit", "recruit_reserve", "recruit_temporary", "emergency_hire"})
RECRUIT_CONFIRM_LABELS = frozenset({"招募", "确认招募", "临时招募", "雇佣", "确认雇佣"})
TICKET_PROFESSIONS = {"先锋": "PIONEER", "近卫": "WARRIOR", "重装": "TANK", "狙击": "SNIPER",
                      "术师": "CASTER", "医疗": "MEDIC", "辅助": "SUPPORT", "特种": "SPECIAL"}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_move(action: ObservedAction) -> bool:
    return action.kind.lower() == "move" or action.metadata.get("operation") == "move"


def _identity_label(value: Any) -> str:
    """Exact client labels with presentational whitespace/markup removed."""
    return re.sub(r"[\s\"“”‘’]", "", re.sub(r"<[^>]*>", "", str(value or "")))


class ObservedMenuMetadataAdapter:
    """Resolve only the identities and quantities actually attached to a card.

    This deliberately does not search the rest of the screen for a convenient
    item name or price. Vision owns card/quantity association. Public catalog
    records describe the identified item; they never generate an offer.
    """

    def __init__(self, catalog, evidence_root: Path | None = None):
        self.catalog = catalog
        self.names: dict[str,set[str]] = {}
        for definition in catalog.items.values():
            self.names.setdefault(_identity_label(definition.name),set()).add(definition.canonical_id)
        evidence = evidence_root or ROOT/"data/evidence"
        source = evidence/"rogue6_observed_operator_catalog_v1.json"
        self.operators = json.loads(source.read_text(encoding="utf-8"))["ordinary_and_exclusive_characters"] if source.is_file() else {}
        tickets = evidence/"rogue6_operator_economy_rules_v1.json"
        self.recruit_tickets = json.loads(tickets.read_text(encoding="utf-8"))["client"]["recruit_tickets"] if tickets.is_file() else {}
        self.operator_names: dict[str,set[str]] = {}
        for identity, record in self.operators.items():
            self.operator_names.setdefault(_identity_label(record["name"]),set()).add(identity)

    def _ticket_metadata(self, metadata):
        """Keep same-name variants unresolved and encode only their shared facts."""
        for key in ("candidate_item_ids", "identity_ambiguous", "ticket_identity_verified",
                    "ticket_professions", "ticket_rarities", "ticket_mechanist_eligible", "ticket_catalog_rarity"):
            metadata.pop(key, None)
        name = metadata.get("ticket_name") or metadata.get("item_name")
        matches = self.names.get(_identity_label(name), set()) if name else set()
        if name and metadata.get("item_name") and _identity_label(name) != _identity_label(metadata["item_name"]):
            metadata["identity_conflict"] = True
        identity = metadata.get("item_id")
        if identity:
            if name and identity not in matches:
                metadata["identity_conflict"] = True
            matches = {identity} if identity in self.catalog.items else set()
        if not matches or any(self.catalog.items[key].category != "RECRUIT_TICKET" for key in matches):
            metadata["identity_unrecognized"] = True
            return
        metadata["ticket_identity_verified"] = True
        metadata["candidate_item_ids"] = sorted(matches)
        metadata["category"] = "RECRUIT_TICKET"
        if len(matches) > 1:
            metadata["identity_ambiguous"] = True
            metadata["identity_source"] = "exact_observed_ticket_name_ambiguous"
        elif not identity:
            metadata["item_id"] = next(iter(matches))
            metadata["identity_source"] = "exact_observed_ticket_name"
        rarities = {self.catalog.items[key].rarity for key in matches}
        if len(rarities) == 1:
            metadata["ticket_catalog_rarity"] = next(iter(rarities))
        records = [self.recruit_tickets.get(key) for key in matches]
        if not all(records):
            metadata["ticket_identity_verified"] = False
            return
        professions = set.intersection(*(set(record["professionList"]) for record in records))
        tiers = set.intersection(*(set(record["rarityList"]) for record in records))
        metadata["ticket_professions"] = sorted(professions)
        metadata["ticket_rarities"] = sorted(tiers)
        if metadata.get("ticket_profession"):
            profession = TICKET_PROFESSIONS.get(_identity_label(metadata["ticket_profession"]))
            if profession is None or profession not in professions:
                metadata["identity_conflict"] = True
        metadata["ticket_mechanist_eligible"] = "TANK" in professions and "TIER_6" in tiers

    def adapt(self, action: ObservedAction) -> dict[str,Any]:
        metadata = dict(action.metadata)
        metadata.pop("operator_identity_verified", None)
        if metadata.get("operation", action.kind) == "select_recruit_ticket":
            self._ticket_metadata(metadata)
        identity = metadata.get("item_id")
        item_name = metadata.get("item_name")
        if not identity and item_name:
            matches = self.names.get(_identity_label(item_name),set())
            if len(matches)==1:
                identity = metadata["item_id"] = next(iter(matches))
                metadata["identity_source"] = "exact_observed_item_name"
            else:
                metadata["identity_unrecognized"] = True
        definition = self.catalog.items.get(identity)
        if definition:
            if item_name and definition.canonical_id not in self.names.get(_identity_label(item_name),set()):
                metadata["identity_conflict"] = True
            metadata["category"] = definition.category
        operator = metadata.get("operator_id")
        if not operator and metadata.get("operator_name"):
            matches = self.operator_names.get(_identity_label(metadata["operator_name"]),set())
            if len(matches)==1:
                operator = metadata["operator_id"] = next(iter(matches))
            else:
                metadata["identity_unrecognized"] = True
        if operator:
            if operator not in self.operators:
                metadata["identity_unrecognized"] = True
            elif metadata.get("operator_name") and operator not in self.operator_names.get(_identity_label(metadata["operator_name"]),set()):
                metadata["identity_conflict"] = True
            else:
                metadata["operator_identity_verified"] = True
            # Existing recruit/temporary/hire option semantics place the actual
            # displayed operator in item_id. Keep ticket identity separately.
            metadata.setdefault("ticket_item_id",identity)
            metadata["item_id"] = operator
        if "uses_remaining" not in metadata and _number(metadata.get("remaining_uses")):
            metadata["uses_remaining"] = metadata["remaining_uses"]
        delta = dict(metadata.get("resource_delta") or {})
        costs = metadata.get("resource_costs",{})
        if isinstance(costs,dict):
            for resource,cost in costs.items():
                if _number(cost) and cost>=0:
                    delta.setdefault(resource,-cost)
        if _number(metadata.get("hope_cost")) and metadata["hope_cost"]>=0:
            delta.setdefault("hope",-metadata["hope_cost"])
        metadata["resource_delta"] = delta
        return metadata


def _recruitment_confirmation_safe(metadata, observation):
    if (observation.scene != "recruitment" or metadata.get("preview_only")
        or metadata.get("selection_stage") != "operator_confirm"
        or metadata.get("grounded") is not True
        or metadata.get("source_frame_id") != observation.frame_id
        or metadata.get("operator_identity_verified") is not True
        or not metadata.get("operator_id")
        or metadata.get("selected_operator_id") != metadata.get("operator_id")
        or metadata.get("button_enabled_observed") is not True):
        return False
    costs = metadata.get("resource_costs", {})
    if not isinstance(costs, dict):
        return False
    costs = dict(costs)
    if "hope_cost" in metadata:
        hope_cost = metadata["hope_cost"]
        if not _number(hope_cost) or ("hope" in costs and costs["hope"] != hope_cost):
            return False
        costs["hope"] = hope_cost
    if not costs:
        return False
    # A free confirmation still needs explicitly observed zero cost and a
    # same-frame balance. No known catalogue/default recruitment price is used.
    return all(_number(cost) and cost >= 0 and _number(observation.resources.get(resource))
               and observation.resources[resource] >= cost for resource, cost in costs.items())


def action_is_safe(action: ObservedAction, observation: LiveObservation, *, threshold: float = 0.85) -> bool:
    """Only mask legality and the requested scope; never score a strategy."""
    from blackflow_rl.policy_constraints import FORBIDDEN_ITEM_FLAGS

    if not action.enabled or not _number(action.confidence) or action.confidence < threshold:
        return False
    if len(action.bbox) != 4 or not all(_number(v) for v in action.bbox):
        return False
    x, y, width, height = action.bbox
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return False
    image_width, image_height = observation.metadata.get("image_width"), observation.metadata.get("image_height")
    if _number(image_width) and x + width > image_width:
        return False
    if _number(image_height) and y + height > image_height:
        return False
    metadata = action.metadata
    if metadata.get("identity_conflict"):
        return False
    if metadata.get("source_frame_id") not in (None, observation.frame_id):
        return False
    kind, operation = action.kind.lower(), str(metadata.get("operation", action.kind)).lower()
    if kind in BLOCKED_OPERATIONS or operation in BLOCKED_OPERATIONS or metadata.get("battle") or metadata.get("starts_battle"):
        return False
    label = action.label.replace(" ", "").lower()
    if any(word.replace(" ", "") in label for word in BLOCKED_LABELS + OTHER_ENDING_LABELS):
        return False
    if metadata.get("ending_id") not in (None, "", "1", 1, "ending_1", "first"):
        return False
    if metadata.get("item_id") in FORBIDDEN_ITEM_FLAGS:
        return False
    if set(metadata.get("add_items", ())) & FORBIDDEN_ITEM_FLAGS:
        return False
    target = next((node for node in observation.nodes if node.node_id == action.target_node_id), None)
    if target is not None and target.node_type in {"STORY", "STORY_HIDDEN", "DOOR"} and not target.completed:
        return False
    if operation == "select_recruit_ticket":
        if (observation.scene != "recruitment" or metadata.get("selection_stage") != "ticket_preview"
            or metadata.get("preview_only") is not True or metadata.get("grounded") is not True
            or metadata.get("source_frame_id") != observation.frame_id
            or metadata.get("ticket_identity_verified") is not True):
            return False
    elif operation in RECRUIT_OPERATIONS or _identity_label(action.label) in RECRUIT_CONFIRM_LABELS:
        if operation not in RECRUIT_OPERATIONS or not _recruitment_confirmation_safe(metadata, observation):
            return False
    if metadata.get("selection_stage") == "operator_preview" or action.kind == "operator_preview":
        if (observation.scene != "recruitment" or operation != "event" or action.kind != "operator_preview"
            or metadata.get("selection_stage") != "operator_preview" or metadata.get("preview_only") is not True
            or metadata.get("grounded") is not True or metadata.get("source_frame_id") != observation.frame_id
            or metadata.get("operator_identity_verified") is not True):
            return False
    if operation == "purchase":
        price = metadata.get("price")
        gold = observation.resources.get("gold")
        if not _number(price) or not _number(gold) or price < 0 or price > gold:
            return False
    if not metadata.get("preview_only"):
        costs = dict(metadata.get("resource_costs") or {})
        if _number(metadata.get("hope_cost")):
            costs["hope"] = metadata["hope_cost"]
        for resource,cost in costs.items():
            quantity = observation.resources.get(resource)
            if not _number(cost) or cost<0:
                return False
            if cost==0:
                continue
            if not _number(quantity) or quantity<cost:
                return False
    return True


class ObservedFeatureEncoder:
    """Current checkpoint feature layout populated exclusively by observations.

    Public catalogue identity/rarity is allowed; sampled prices, loot, hidden
    maps and simulator-derived legal actions are deliberately unavailable.
    """

    def __init__(self, encoder: Any):
        self.layout = encoder
        self.metadata_adapter = ObservedMenuMetadataAdapter(encoder.simulator.economy.catalog)
        self.last_missing_fields: tuple[str, ...] = ()
        self.last_coverage: dict[str,Any] = {}

    def metadata_for(self, action):
        return self.metadata_adapter.adapt(action)

    def _shop_features(self, observation, node_id, options):
        """Populate only a fully observed current shelf, never cached stock."""
        import numpy as np
        from blackflow_rl.features import SHOP_SCALAR_FIELDS, SHOP_CATEGORIES
        layout = self.layout
        result = np.zeros(layout.shop_node_dim,dtype=np.float32)
        shop = observation.metadata.get("shop_state",{})
        if observation.scene!="shop" or not shop.get("shelf_complete") or shop.get("node_id")!=node_id:
            return result
        if shop.get("source_frame_id") not in (None,observation.frame_id):
            return result
        result[0]=1
        for key,index,scale in (("refreshes",1,4),("remaining_refreshes",2,4),("visits",3,10),
                ("sold_parts",4,10),("sold_slots",6,10),("lost_slots",7,10),("total_withdrawn",8,20),("entry_withdrawn",9,12)):
            if _number(shop.get(key)):
                result[index]=shop[key]/scale
        result[5]=bool(shop.get("trade_bonus_paid"))
        categories = np.zeros(len(SHOP_CATEGORIES),dtype=np.float32)
        category_prices = categories.copy()
        identities = np.zeros(len(layout.item_identities),dtype=np.float32)
        item_prices = identities.copy()
        seen_slots=set()
        for action in options:
            row=self.metadata_for(action)
            if row.get("operation")!="purchase" or not _number(row.get("price")):
                continue
            slot=row.get("slot_id",action.action_id)
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            category=row.get("category")
            if category in SHOP_CATEGORIES:
                index=SHOP_CATEGORIES.index(category)
                category_prices[index]=min(category_prices[index],row["price"]) if categories[index] else row["price"]
                categories[index]+=1
            definition=layout.simulator.economy.catalog.items.get(row.get("item_id"))
            identity=layout._item_identity_index.get(definition.canonical_id) if definition else None
            if identity is not None:
                item_prices[identity]=min(item_prices[identity],row["price"]) if identities[identity] else row["price"]
                identities[identity]+=1
        result[len(SHOP_SCALAR_FIELDS):]=np.concatenate((categories/10,category_prices/30,identities/10,item_prices/30))
        return result

    def encode(self, observation: LiveObservation, legal: tuple[ObservedAction, ...]):
        import numpy as np
        from blackflow_rl.features import (EncodedState, INVENTORY_FLAGS, OBSERVED_NODE_LABELS,
            RESOURCE_FIELDS, RESOURCE_SCALES, ITEM_CATEGORIES, OPTION_OPERATIONS, CHOICE_ID_BITS)

        layout = self.layout
        max_nodes, max_options = layout.ruleset.max_nodes, layout.ruleset.max_options
        if len(observation.nodes) > max_nodes:
            raise ValueError("Observed map exceeds checkpoint node capacity")
        options = tuple(a for a in observation.actions if not _is_move(a))
        if len(options) > max_options:
            raise ValueError("Observed menu exceeds checkpoint option capacity")
        node_features = np.zeros((max_nodes, layout.node_feature_dim), dtype=np.float32)
        option_features = np.zeros((max_options, layout.option_feature_dim), dtype=np.float32)
        globals_ = np.zeros(layout.global_feature_dim, dtype=np.float32)
        adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        node_mask = np.zeros(max_nodes, dtype=bool)
        option_mask = np.zeros(max_options, dtype=bool)
        action_mask = np.zeros(max_nodes + max_options, dtype=bool)
        node_index = {node.node_id: index for index, node in enumerate(observation.nodes)}
        if len(node_index) != len(observation.nodes):
            raise ValueError("Observed map repeats a node identity")
        action_index: dict[str, int] = {}
        legal_ids = {a.action_id for a in legal}
        row_max = max((n.row for n in observation.nodes), default=1)
        col_max = max((n.col for n in observation.nodes), default=1)
        # These are fixed normalization denominators in the trained feature
        # contract, not the player's current resources.
        base_ap = layout.ruleset.floor(observation.floor).action_points if observation.floor else 1
        frontier = {a.target_node_id: a for a in legal if _is_move(a)}
        for node in observation.nodes:
            index = node_index[node.node_id]
            node_mask[index] = True
            label = node.node_type if node.node_type in OBSERVED_NODE_LABELS else "UNKNOWN_MYSTERY"
            if not node.revealed:
                label = "UNKNOWN_FEROCITY" if label.startswith("BATTLE_") else "UNKNOWN_MYSTERY"
            node_features[index, OBSERVED_NODE_LABELS.index(label)] = 1
            offset = len(OBSERVED_NODE_LABELS)
            move = frontier.get(node.node_id)
            movement_cost = move.metadata.get("movement_cost") if move else None
            node_features[index, offset:offset+12] = (
                node.node_id == observation.current_node_id, node.completed, node.revealed,
                node.revealed and label in {"FINAL", "EVACUATE", "BATTLE_BOSS"},
                observation.metadata.get("pending_node_id") == node.node_id,
                move is not None, node.row/max(1, row_max), node.col/max(1, col_max),
                0, movement_cost/base_ap if _number(movement_cost) else 0,
                bool(options) and observation.metadata.get("pending_node_id") == node.node_id,
                label.startswith("BATTLE_") and label != "BATTLE_SHOP",
            )
            if label in {"SCRAP_SHOP","BATTLE_SHOP"}:
                node_features[index,offset+layout.NODE_SCALAR_DIM:] = self._shop_features(observation,node.node_id,options)
        for left, right in observation.edges:
            if left not in node_index or right not in node_index:
                raise ValueError("Observed edge has an unobserved endpoint")
            adjacency[node_index[left], node_index[right]] = 1
            adjacency[node_index[right], node_index[left]] = 1
        for action in observation.actions:
            if _is_move(action) and action.target_node_id in node_index:
                index = node_index[action.target_node_id]
                action_index[action.action_id] = index
                action_mask[index] = action.action_id in legal_ids
        operation_start = len(RESOURCE_FIELDS) + 4 + len(ITEM_CATEGORIES)
        choice_start = operation_start + len(OPTION_OPERATIONS)
        physical_start = layout.option_feature_dim - len(layout.item_identities) - 12
        for index, action in enumerate(options):
            row = option_features[index]
            metadata = self.metadata_for(action)
            operation = str(metadata.get("operation", action.kind)).lower()
            option_mask[index] = True
            action_index[action.action_id] = max_nodes + index
            action_mask[max_nodes + index] = action.action_id in legal_ids
            delta = metadata.get("resource_delta", {})
            for resource_index, field in enumerate(RESOURCE_FIELDS):
                if _number(delta.get(field)):
                    row[resource_index] = delta[field] / RESOURCE_SCALES[resource_index]
            row[len(RESOURCE_FIELDS)+1] = len(metadata.get("add_items", ())) / 3
            row[len(RESOURCE_FIELDS)+2] = len(metadata.get("remove_items", ())) / 3
            row[len(RESOURCE_FIELDS)+3] = float(action.enabled)
            definition = layout.simulator.economy.catalog.items.get(metadata.get("item_id"))
            category = definition.category if definition else metadata.get("category")
            if category in ITEM_CATEGORIES:
                row[len(RESOURCE_FIELDS)+4+ITEM_CATEGORIES.index(category)] = 1
            row[operation_start + OPTION_OPERATIONS.index(operation if operation in OPTION_OPERATIONS else "unknown")] = 1
            choice = layout._client_choice_index.get(metadata.get("choice_id"), 0)
            row[choice_start:choice_start+CHOICE_ID_BITS] = [(choice >> bit) & 1 for bit in range(CHOICE_ID_BITS)]
            for field, position, divisor in (("price",0,30), ("quantity",1,5), ("uses_remaining",5,10), ("appraisal",6,30)):
                if _number(metadata.get(field)):
                    row[physical_start+position] = metadata[field]/divisor
            row[physical_start+7] = bool(metadata.get("ends_node"))
            if _number(metadata.get("parts_capacity_delta")):
                row[physical_start+8] = metadata["parts_capacity_delta"]/5
            elif operation=="parts_capacity":
                row[physical_start+8] = 1/5
            row[physical_start+9] = metadata.get("item_id")=="char_4230_mcnist"
            if operation=="select_recruit_ticket":
                from blackflow_rl.operator_economy import mechanist_eligible_ticket_ids
                row[physical_start+10] = (metadata.get("item_id") in mechanist_eligible_ticket_ids()
                                          or metadata.get("ticket_mechanist_eligible") is True)
            row[physical_start+11] = category=="UPGRADE_TICKET"
            if definition:
                row[physical_start+2] = {"NORMAL":1,"RARE":2,"SUPER_RARE":3,"SPECIAL":4}.get(definition.rarity,0)/4
                # Catalogue prices remain properties; the actual purchase
                # price above must independently have been observed.
                row[physical_start+3] = (definition.base_buy_price or 0)/30
                row[physical_start+4] = (definition.sell_price or 0)/30
                row[-len(layout.item_identities):] = layout._item_identity_features(metadata.get("item_id"))
            elif metadata.get("ticket_identity_verified"):
                # An ambiguous ordinary/candle title has no exact identity
                # one-hot. Only properties shared by every matching ticket may
                # fill the existing frozen feature layout.
                row[physical_start+2] = {"NORMAL":1,"RARE":2,"SUPER_RARE":3,"SPECIAL":4}.get(metadata.get("ticket_catalog_rarity"),0)/4

        resources = observation.resources
        if observation.floor:
            globals_[0] = (observation.floor-1)/4
        globals_[1] = sum(n.completed for n in observation.nodes)/max(1,len(observation.nodes))
        if _number(resources.get("action_points")):
            globals_[2] = resources["action_points"]/base_ap
        if _number(resources.get("hp")) and _number(resources.get("max_hp")) and resources["max_hp"] > 0:
            globals_[3] = resources["hp"]/resources["max_hp"]
        for name, index, divisor in (("max_hp",4,20),("shield",5,20),("gold",6,50),("hope",7,30),
                                    ("parts",8,10),("relics",9,10),("tickets",10,10)):
            if _number(resources.get(name)):
                globals_[index] = resources[name]/divisor
        # team_strength is a synthetic training statistic, never invented for
        # real operators. Its reserved input stays unobserved.
        globals_[12] = observation.scene != "map" and bool(options)
        inventory = observation.metadata.get("inventory", ())
        for index, flag in enumerate(INVENTORY_FLAGS):
            globals_[layout.GLOBAL_BASE_DIM+index] = flag in inventory
        items = observation.metadata.get("items", ())
        economy_offset = layout.GLOBAL_BASE_DIM + len(INVENTORY_FLAGS) + layout.inventory_detail_dim
        globals_[economy_offset+6] = 1  # this feature identifies the item-economy UI contract
        capacity = resources.get("parts_capacity")
        if _number(capacity):
            globals_[economy_offset+7] = capacity/20
            if _number(resources.get("parts")):
                globals_[economy_offset+8] = (capacity-resources["parts"])/20
        for index, category in enumerate(ITEM_CATEGORIES):
            globals_[economy_offset+2+index] = sum(item.get("category")==category for item in items)/20
        for field,index,divisor in (("supply_vouchers",9,10),("commander_level",20,10),("commander_exp",21,100),
                ("bank_balance",46,1000),("total_bank_withdrawn",47,100),("bank_balance_spent",48,1000)):
            if _number(resources.get(field)):
                globals_[economy_offset+index] = resources[field]/divisor
        # All lists below are observed snapshots. Candidate cards do not enter
        # formal inventory or operator counts until a subsequent screenshot.
        for field,index,divisor in (("formal_operator_ids",39,6),("available_operator_ids",40,6),("promoted_operator_ids",41,6),
                ("pending_recruit_ticket_ids",50,3),("stored_recruit_ticket_ids",51,3),("temporary_recruit_offers",63,3)):
            if field in observation.metadata:
                globals_[economy_offset+index] = len(observation.metadata[field])/divisor
        if "available_operator_ids" in observation.metadata:
            globals_[economy_offset+44] = "char_4230_mcnist" in observation.metadata["available_operator_ids"]
        if "promoted_operator_ids" in observation.metadata:
            globals_[economy_offset+45] = "char_4230_mcnist" in observation.metadata["promoted_operator_ids"]
        globals_[economy_offset+10] = any(item.get("equipped") for item in items)
        globals_[economy_offset+11] = sum(item.get("uses_remaining",0) or 0 for item in items if item.get("category")=="MOVE")/30
        globals_[economy_offset+12] = sum(item.get("appraisal",0) or 0 for item in items if item.get("category")=="GOODS")/50
        shop=observation.metadata.get("shop_state",{})
        if observation.scene=="shop" and shop.get("source_frame_id") in (None,observation.frame_id):
            for field,index,divisor in (("refreshes",13,10),("sold_parts",14,10),("entry_withdrawn",49,12)):
                if _number(shop.get(field)):
                    globals_[economy_offset+index]=shop[field]/divisor
            globals_[economy_offset+15]=bool(shop.get("trade_bonus_paid"))
        for item in items:
            globals_[-len(layout.item_identities):] += layout._item_identity_features(item.get("item_id"))
        move_offset = layout.GLOBAL_BASE_DIM+len(INVENTORY_FLAGS)
        for item_id in layout.move_identities:
            held = [item for item in items if item.get("item_id")==item_id]
            for uses in layout.move_use_bins:
                globals_[move_offset] = sum(item.get("uses_remaining")==uses for item in held)/20
                move_offset += 1
            globals_[move_offset] = sum(item.get("uses_remaining",0) for item in held if item.get("equipped"))/10
            globals_[move_offset+1] = sum(bool(item.get("expires_on_floor_change")) for item in held)/20
            move_offset += 2
        for item_id in layout.goods_identities:
            values = [item["appraisal"] for item in items if item.get("item_id")==item_id and _number(item.get("appraisal"))]
            globals_[move_offset:move_offset+4] = (sum(values)/50,min(values,default=0)/50,max(values,default=0)/50,sum(min(max(0,v),2) for v in values)/20)
            move_offset += 4
        self.last_missing_fields = tuple(name for name in RESOURCE_FIELDS if not _number(resources.get(name)))
        self.last_coverage = {
            "source_frame_id":observation.frame_id,
            "missing_resource_fields":self.last_missing_fields,
            "inventory_complete":bool(observation.metadata.get("inventory_complete",False)),
            "identified_item_actions":sum(bool(self.metadata_for(a).get("item_id")) for a in options),
            "displayed_action_count":len(options),
            "operator_embedding_scope":"mechanist_identity_and_observed_costs_only",
            "ticket_embedding_scope":"operation_shared_rarity_and_mechanist_eligibility_only",
            "ambiguous_ticket_actions":sum(bool(self.metadata_for(a).get("identity_ambiguous")) for a in options),
            "cross_frame_resource_reuse":False,
        }
        return EncodedState(node_features, adjacency, node_mask, globals_, option_features, option_mask, action_mask), action_index, node_index


class CurrentNeuralPolicy:
    """Use the locally selected route and menu networks, with no fallback."""

    def __init__(self, root: str | Path | None = None, selection_path: str | Path | None = None, *, confidence_threshold: float = 0.85):
        self.root = Path(root) if root is not None else ROOT
        self.selection_path = Path(selection_path) if selection_path else self.root/"data/policies/current_neural_controller.json"
        self.confidence_threshold = confidence_threshold
        self.ready = False
        self.load_error: str | None = None
        self.selection: dict[str, Any] = {}
        self._attempted_load = False

    @property
    def name(self) -> str:
        return self.selection.get("name", "current_neural_controller")

    def readiness(self) -> dict[str,Any]:
        """Self-contained capability report for logs/UI; no success projection."""
        return {"ready":self.ready,"load_error":self.load_error,"policy_name":self.name,
            "first_ending_only":True,"battle_control":False,"simulation_used":False,
            "last_observation_coverage":getattr(getattr(self,"encoder",None),"last_coverage",{}),
            "limitations":["Unobserved inventory, counters and shop history are not reconstructed.",
                "The frozen network has no general per-operator identity embedding.",
                "The frozen layout has no ticket category, profession or exact identity embedding; only ticket operation, shared rarity and mechanist eligibility are encoded.",
                "Purchases need an observed actual price and balance; costs are never sampled.",
                "No complete real-game first-ending run has been established by these unit tests."]}

    def _artifact(self, field: str) -> Path:
        path = (self.root / self.selection[field]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"Selected {field} escapes the repository")
        if not path.is_file():
            raise FileNotFoundError(f"Selected {field} is missing: {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != self.selection[field+"_sha256"]:
            raise ValueError(f"Selected {field} checksum differs")
        return path

    def load(self) -> bool:
        self._attempted_load = True
        try:
            import torch
            from blackflow_rl.catalog import load_catalog
            from blackflow_rl.rules import load_ruleset
            from blackflow_rl.features import FeatureEncoder
            from blackflow_rl.policy_constraints import UserPolicyConstraints
            from blackflow_rl.network import GraphPolicyValueNetwork, NetworkConfig
            from blackflow_rl.neural_macro_route import MacroRouteNetwork, MacroRouteController, implementation_sha256

            self.selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
            if self.selection.get("schema_version") != 1 or self.selection.get("kind") != "macro":
                raise ValueError("Unsupported current neural selection")
            route_path, menu_path = self._artifact("checkpoint"), self._artifact("menu_checkpoint")
            self._artifact("profile")
            # These are hash-verified local training artifacts containing
            # historical Python state. Only weight dictionaries are used.
            route = torch.load(route_path, map_location="cpu", weights_only=False)
            menu = torch.load(menu_path, map_location="cpu", weights_only=False)
            if route.get("algorithm") != MacroRouteController.ALGORITHM or route.get("implementation_sha256") != implementation_sha256():
                raise ValueError("Selected route implementation differs")
            if menu.get("algorithm") != "PPO_DIRECT_NEURAL" or menu.get("format_version") != 1:
                raise ValueError("Selected menu checkpoint format differs")
            if route.get("base_sha256") != self.selection["menu_checkpoint_sha256"]:
                raise ValueError("Selected route does not reference the selected frozen menu")
            if any(payload.get("environment_sha256") != self.selection["environment_sha256"] for payload in (route, menu)):
                raise ValueError("Selected checkpoints have inconsistent environment contracts")
            if menu.get("network_implementation_sha256") != sha256((self.root/"blackflow_rl/network.py").read_bytes()).hexdigest():
                raise ValueError("Selected network implementation differs")
            rules = load_ruleset()
            layout_context = SimpleNamespace(ruleset=rules,economy=SimpleNamespace(catalog=load_catalog()),action_size=rules.max_nodes+rules.max_options)
            constraints = UserPolicyConstraints(**menu["policy_constraints"])
            if not constraints.first_ending_only:
                raise ValueError("Selected policy is not restricted to the first ending")
            layout = FeatureEncoder(layout_context, constraints=constraints)
            if any(payload.get("feature_schema_sha256") != layout.schema_sha256 for payload in (route, menu)):
                raise ValueError("Selected feature schema differs")
            torch.set_num_threads(1)
            self.menu_model = GraphPolicyValueNetwork(NetworkConfig(**menu["network_config"]))
            self.menu_model.load_state_dict(menu["model_state_dict"], strict=True)
            self.route_model = MacroRouteNetwork(self.menu_model, len(layout.move_identities))
            self.route_model.load_state_dict(route["route_state_dict"], strict=True)
            self.menu_model.eval()
            self.route_model.eval()
            self.encoder = ObservedFeatureEncoder(layout)
            self.ready, self.load_error = True, None
        except Exception as exc:
            self.ready, self.load_error = False, f"{type(exc).__name__}: {exc}"
        return self.ready

    def _refuse(self, reason: str) -> PolicyDecision:
        return PolicyDecision(None, reason, self.name)

    def select(self, observation: LiveObservation) -> PolicyDecision:
        if observation.ending_first_confirmed:
            return self._refuse("first_ending_confirmed")
        if observation.floor is not None and observation.floor not in range(1,6):
            return self._refuse("route_outside_first_ending_scope")
        if observation.scene in BLOCKED_SCENES:
            return self._refuse("manual_battle_required" if observation.scene in {"battle", "battle_start", "combat"} else "unrecognized_or_terminal_scene")
        if not _number(observation.confidence) or observation.confidence < self.confidence_threshold:
            return self._refuse("observation_confidence_too_low")
        if not self._attempted_load:
            self.load()
        if not self.ready:
            return self._refuse("neural_weights_unavailable: "+str(self.load_error))
        if len({a.action_id for a in observation.actions}) != len(observation.actions):
            return self._refuse("duplicate_observed_action_id")
        legal = tuple(a for a in observation.actions if action_is_safe(
            replace(a,metadata=self.encoder.metadata_for(a)), observation, threshold=self.confidence_threshold))
        if not legal:
            return self._refuse("no_grounded_scope_legal_action")
        routing = observation.scene == "map" and any(_is_move(a) for a in legal)
        if routing:
            if observation.floor not in range(1, 6) or observation.current_node_id not in {n.node_id for n in observation.nodes}:
                return self._refuse("route_requires_observed_floor_and_current_node")
            missing = [field for field in REQUIRED_ROUTE_RESOURCES if not _number(observation.resources.get(field))]
            if missing:
                return self._refuse("route_missing_observed_resources: "+",".join(missing))
            resources = observation.resources
            if any(resources[field]<0 for field in REQUIRED_ROUTE_RESOURCES) or resources["max_hp"]<=0 or resources["hp"]>resources["max_hp"]:
                return self._refuse("route_invalid_observed_resources")
            legal = tuple(a for a in legal if not _is_move(a) or a.target_node_id in {n.node_id for n in observation.nodes})
        else:
            from blackflow_rl.features import OPTION_OPERATIONS
            legal = tuple(a for a in legal if str(a.metadata.get("operation", a.kind)).lower() in set(OPTION_OPERATIONS)-{"unknown", "needs_observation"})
        if not legal:
            return self._refuse("no_recognized_action_semantics")
        try:
            import torch
            encoded, indices, nodes = self.encoder.encode(observation, legal)
            tensors = {name: value.unsqueeze(0) for name,value in encoded.as_torch().items()}
            legal = tuple(action for action in legal if action.action_id in indices)
            if not legal:
                return self._refuse("no_encoded_observed_action")
            with torch.inference_mode():
                if routing:
                    logits = self.route_model(**tensors, **self._macro_inputs(observation, legal, indices, nodes))[0]
                else:
                    logits, _ = self.menu_model(**tensors)
                    logits = logits[0, [indices[a.action_id] for a in legal]]
                if not torch.isfinite(logits).all():
                    return self._refuse("nonfinite_neural_output")
                probabilities = torch.softmax(logits, dim=0)
                selected = int(logits.argmax())
                confidence = float(probabilities[selected])
            return PolicyDecision(legal[selected], "selected_from_observed_legal_actions", self.name, True, confidence)
        except Exception as exc:
            return self._refuse("observation_encoding_failed: "+str(exc))

    def _macro_inputs(self, observation, actions, indices, nodes):
        import torch
        from blackflow_rl.neural_macro_route import FACTS

        count = len(actions)
        facts = torch.zeros((1,count,len(FACTS)))
        node_ids = torch.zeros((1,count),dtype=torch.long)
        option_ids = node_ids.clone()
        gear_ids = node_ids.clone()
        current = next(node for node in observation.nodes if node.node_id == observation.current_node_id)
        by_id = {node.node_id:node for node in observation.nodes}
        items = observation.metadata.get("items", ())
        vehicles = [item for item in items if item.get("category")=="MOVE"]
        p05 = sum(item.get("item_id", "").endswith("P_05") for item in items)
        p06 = sum(item.get("item_id", "").endswith("P_06") for item in items)
        gear_vocabulary = {item:i+1 for i,item in enumerate(self.encoder.layout.move_identities)}
        for index,action in enumerate(actions):
            metadata = action.metadata
            move = _is_move(action)
            gear = metadata.get("equipment_item_id")
            random_move = bool(metadata.get("random_move"))
            node = by_id.get(action.target_node_id) if move else None
            known = bool(node and node.revealed and not random_move)
            cost = metadata.get("movement_cost")
            uses = metadata.get("uses_remaining")
            facts[0,index] = torch.tensor((
                float(move),float(not move),float(bool(gear)),float(metadata.get("same_equipment",True)),float(random_move),
                cost/5 if _number(cost) else 0,uses/10 if _number(uses) else 0,float(bool(metadata.get("expires_on_floor_change"))),
                (node.row-current.row)/10 if node and not random_move else 0,
                (node.col-current.col)/10 if node and not random_move else 0,
                p05/10 if known and node.node_type=="WISH" and gear else 0,
                2*p06/10 if known and node.node_type=="SACRIFICE" and gear else 0,
                float(bool(node and node.completed and not random_move)),float(known),p05/10,p06/10,
                observation.floor/5,observation.resources["action_points"]/20,len(vehicles)/10,
                sum(item.get("uses_remaining",0) or 0 for item in vehicles)/20,
            ))
            node_ids[0,index] = nodes[action.target_node_id] if node else 0
            option_ids[0,index] = max(0,indices[action.action_id]-self.encoder.layout.ruleset.max_nodes) if not move else 0
            gear_ids[0,index] = gear_vocabulary.get(gear,0)
        return dict(macro_facts=facts,macro_nodes=node_ids,macro_options=option_ids,macro_gears=gear_ids,macro_mask=torch.ones((1,count),dtype=torch.bool))
