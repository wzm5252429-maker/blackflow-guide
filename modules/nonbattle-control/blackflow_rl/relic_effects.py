"""Client-driven economic hooks, without pretending to simulate combat buffs.

The engine calls these at actual triggers. They do not repeat base battle
rewards, floor AP setup, gold multipliers, shop entry squad rewards or concept
rewards. Unknown rounding, random laws and acquisition ordering remain marked.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .domain import ResourceDelta


RESOURCE_TARGETS = {
    "rogue_6_gold": "gold", "rogue_6_hp": "hp", "rogue_6_hpmax": "max_hp",
    "rogue_6_population": "hope", "rogue_6_shield": "shield", "rogue_6_ap": "action_points",
}
PART_TYPES = frozenset({"MOVE", "GOODS", "PASSIVE"})


def blackboard(buff):
    return {row["key"]: row["valueStr"] if row.get("valueStr") is not None else row.get("value", 0)
            for row in buff.get("blackboard", ())}


def owned_relics(engine, state):
    seen = set()
    for item in state.item_instances:
        if item.category != "RELIC":
            continue
        definition = engine.catalog.variant(item.item_id, difficulty=engine.config.difficulty)
        if definition.canonical_id not in seen:
            seen.add(definition.canonical_id)
            yield definition


def _reward(engine, state, target, count, source):
    count = int(count)
    if not count:
        return state
    if target in RESOURCE_TARGETS:
        return engine.apply(state, ResourceDelta(**{RESOURCE_TARGETS[target]: count}), source)
    if target == "rogue_6_max_weight":
        return replace(state, parts_capacity=state.parts_capacity + count)
    if target == "rogue_6_squad_capacity":
        return replace(state, squad_capacity=state.squad_capacity + count)
    if target in engine.catalog.observed_pools:
        return engine.grant(state, "pool:" + target, count, source=source)
    if target in engine.catalog.items:
        definition = engine.catalog[target]
        if definition.category in ("RECRUIT_TICKET", "UPGRADE_TICKET"):
            # Preserve ticket identity in the ledger/inventory while retaining
            # the existing abstract count. Recruiting still needs a real choice.
            state = engine.apply(state, ResourceDelta(tickets=count), source)
        return engine.grant(state, target, count, source=source)
    return engine.entry(state, "unimplemented_client_reward", quantity=count, source=source + ":" + str(target))


def apply_immediate(engine, state, definition):
    """Replace (do not additionally call) EconomyEngine.relic_immediate."""
    definition = engine.catalog.variant(definition.item_id, difficulty=engine.config.difficulty)
    for buff in definition.client_buffs:
        bb = blackboard(buff)
        if buff["key"] == "immediate_reward":
            state = _reward(engine, state, bb.get("id"), bb.get("count", 0), definition.item_id)
        elif buff["key"] == "scrap_fill_up":
            for _ in range(max(0, state.parts_capacity - state.resources.parts)):
                state = _reward(engine, state, bb["id"], 1, definition.item_id)
        elif buff["key"] == "level_char_limit_add":
            key = "relic:deployment_capacity_add"
            state = engine.set_counter(state, key, engine.counter(state, key) + int(bb["value"]))
    return on_collection_changed(engine, state)


def on_collection_changed(engine, state):
    for definition in tuple(owned_relics(engine, state)):
        for buff in definition.client_buffs:
            if buff["key"] != "immediate_reward_on_collection_complete":
                continue
            bb = blackboard(buff)
            if not all(item_id in state.inventory for item_id in bb["ids"].split(",")):
                continue
            key = "collection_reward:" + bb["id"]
            if not engine.counter(state, key):
                state = engine.set_counter(state, key, 1)
                state = _reward(engine, state, bb["id"], bb.get("count", 1), definition.item_id)
    return state


def on_scrap_acquired(engine, state, acquired, *, include_new_item=False):
    """Call after inserting one item, before G_07 spawns its three balls.

    Only pre-existing natural objects react by default. Whether the newly
    obtained vine/moss reacts to its own acquisition is NOT specified in the
    client table; callers can replay an observed include_new_item value. This
    conservative ordering must remain a declared assumption, not a true rule.
    The three child balls each subsequently call this hook as separate obtains.
    """
    if acquired.category not in PART_TYPES:
        return state
    values = []
    for item in state.item_instances:
        eligible = include_new_item or item.instance_id != acquired.instance_id
        delta = (4 if item.item_id == "rogue_6_scrap_G_09" else
                 1 if item.item_id == "rogue_6_scrap_G_07" else 0) if eligible else 0
        values.append(replace(item, appraisal=item.appraisal + delta))
    state = replace(state, item_instances=tuple(values))
    if acquired.item_id in ("rogue_6_scrap_G_07", "rogue_6_scrap_G_09"):
        state = engine.entry(state, "model_assumption", item=acquired,
            source="natural_acquisition:self_trigger=" + str(include_new_item).lower())
    return state


def on_recruit(engine, state, *, recruited_count=1):
    """Grow held corn after actual recruitment or user-confirmed successful emergency hiring."""
    if recruited_count < 0:
        raise ValueError("recruited_count cannot be negative")
    return replace(state, item_instances=tuple(
        replace(item, appraisal=item.appraisal + 3 * recruited_count)
        if item.item_id == "rogue_6_scrap_G_04" else item for item in state.item_instances))


def on_move_appraisal(engine, state, *, wave_deltas: Mapping[str, int] | None = None):
    """Supply observed integer deltas, or explicitly opt into a research prior.

    This replaces the engine's G_10/G_05 valuation loop, not its move effects.
    """
    wave_deltas = {} if wave_deltas is None else wave_deltas
    from .wave_model import OBSERVED_DELTAS, sample_delta
    waves = {item.instance_id for item in state.item_instances if item.item_id == 'rogue_6_scrap_G_05'}
    if set(wave_deltas) - waves:
        raise ValueError('observed wave delta references an unowned wave instance')
    if any(type(delta) is not int or delta not in OBSERVED_DELTAS for delta in wave_deltas.values()):
        raise ValueError('observed wave change must be an integer in the documented -8..11 range')
    tail_probability = engine.config.synthetic_wave_tail_probability
    items = []
    for item in state.item_instances:
        value = item.appraisal
        if item.item_id == "rogue_6_scrap_G_10":
            value = max(0, value - 2)
        elif item.item_id == "rogue_6_scrap_G_05":
            delta = wave_deltas.get(item.instance_id)
            if delta is None and tail_probability is not None:
                rng, state = engine.roll(state, 'wave_appraisal:'+item.instance_id)
                delta = sample_delta(rng, tail_probability)
                # Quantity counts items and must remain nonnegative. The signed
                # appraisal change is metadata, never gold or item acquisition.
                state = engine.entry(state, 'model_assumption', item=item,
                    source='wave:video_integer_prior:tail_mass='+str(tail_probability)
                        +':independent_instances:delta='+str(delta))
            if delta is None:
                state = engine.entry(state, "unresolved_random_outcome", item=item,
                    source="wave:unobserved_integer_delta")
            else:
                value += delta
            # The public 0..999 boundary remains known even when this move's
            # random change is unobserved. Do not preserve an invalid legacy value.
            value = max(0, min(999, value))
        items.append(replace(item, appraisal=value))
    return replace(state, item_instances=tuple(items))


def on_vehicle_exhausted(engine, state):
    """Only a last use triggers this; selling/discarding/floor expiry does not."""
    for definition in tuple(owned_relics(engine, state)):
        for buff in definition.client_buffs:
            if buff["key"] == "immediate_reward_on_vehicle_broken":
                bb = blackboard(buff)
                state = _reward(engine, state, bb["id"], bb.get("count", 1), definition.item_id)
    return state


def _layer_reward(engine, state, definition, *, trigger, increment):
    key = "relic_layer:" + definition.canonical_id
    layer = engine.counter(state, key) + int(increment)
    state = engine.set_counter(state, key, layer)
    for buff in definition.client_buffs:
        if buff["key"] != "relic_layer_reward":
            continue
        bb = blackboard(buff)
        paid_key = key + ":paid:" + str(bb.get("id"))
        if bb.get("trig") == trigger and layer >= int(bb.get("target", 0)):
            if engine.counter(state, paid_key) < int(bb.get("limit", 1)):
                state = engine.set_counter(state, paid_key, engine.counter(state, paid_key) + 1)
                state = _reward(engine, state, bb["id"], bb.get("count", 1), definition.item_id)
    return state


def on_node_enter(engine, state, node):
    """Client layer_node_in triggers; repeat-entry semantics follow the buff key."""
    for definition in tuple(owned_relics(engine, state)):
        for buff in definition.client_buffs:
            if buff["key"] != "layer_node_in":
                continue
            bb = blackboard(buff)
            if node.node_type.value in bb.get("node", "").split(","):
                state = _layer_reward(engine, state, definition, trigger="NODE_INTO", increment=1)
    return state


def on_battle(engine, state, *, perfect: bool | None, won=True, stage_id=None, chase=False):
    """After resolution, before collecting loot: natural values and layer prizes.

    G_06 depends on perfect outcome, never on elite-vs-normal node type. If an
    outcome is unobserved, preserve that uncertainty instead of assuming elite
    battles are perfect or non-elite battles cannot appreciate the tree.
    """
    valued = []
    for item in state.item_instances:
        if item.item_id == "rogue_6_scrap_G_06" and (perfect is False or not won):
            state = engine.remove(state, item.instance_id, "destroy", "frost_tree:nonperfect_battle")
            continue
        amount = 2 if item.item_id == "rogue_6_scrap_G_02" else 0
        if item.item_id == "rogue_6_scrap_G_06":
            if perfect is True and won:
                amount = 4
            elif perfect is None:
                state = engine.entry(state, "unresolved_battle_outcome", item=item, source="frost_tree:perfect_unknown")
        valued.append(replace(item, appraisal=item.appraisal + amount))
    state = engine.sync(replace(state, item_instances=tuple(valued)))
    if not won:
        return state
    state = engine.audit_notebook_battle(state,stage_id=stage_id)
    for definition in tuple(owned_relics(engine, state)):
        for buff in definition.client_buffs:
            if buff["key"] != "layer_pass_stage":
                continue
            bb = blackboard(buff)
            matches = stage_id is not None and stage_id in bb.get("stage_ids", "").split(",")
            # The client usage explicitly names one completed chase, even when
            # the caller lacks the exact ro6_c_* stage identity.
            if matches or chase and definition.canonical_id == "rogue_6_relic_artifact_3":
                state = _layer_reward(engine, state, definition, trigger="AFTER_BATTLE", increment=1)
    return state


def on_commander_level(engine, state, level):
    """Level is externally observed/calculated; no fabricated XP progression."""
    if not 1 <= level <= 10:
        raise ValueError("commander level must be 1..10")
    state = engine.set_counter(state, "commander_level", level)
    for definition in tuple(owned_relics(engine, state)):
        for buff in definition.client_buffs:
            if buff["key"] != "player_level_rewards":
                continue
            bb = blackboard(buff)
            paid = "level_reward:" + definition.canonical_id
            if level >= int(bb["level"]) and not engine.counter(state, paid):
                state = engine.set_counter(state, paid, 1)
                state = _reward(engine, state, bb["id"], bb.get("count", 1), definition.item_id)
    return state


def on_floor_enter(engine, state, *, zone_id=None):
    """Apply actual zone-entry resources, and create a marked natural reward."""
    zone_id = zone_id or f"zone_{state.floor}"
    for definition in tuple(owned_relics(engine, state)):
        for buff in definition.client_buffs:
            bb = blackboard(buff)
            if buff["key"] == "zone_into_reward" and bb.get("zone") == zone_id:
                state = _reward(engine, state, bb["id"], bb.get("count", 1), definition.item_id)
            elif buff["key"] == "zone_into_node_attach_buff" and bb.get("id") == "rogue_6_bubble_07":
                kinds = set(bb.get("node", "").split(","))
                candidates = [node for node in state.floor_map.nodes if node.node_type.value in kinds]
                if candidates:
                    rng, state = engine.roll(state, "hide_and_seek_marker")
                    selected = rng.choice(candidates)
                    state = engine.set_counter(state, "hidden_natural:" + selected.node_id, 1)
                    state = engine.entry(state, "model_assumption", source="hide_and_seek:uniform_marked_node")
                    # The marker is public independently of its underlying
                    # node. It neither reveals that node type nor discovers
                    # a node for grass appraisal.
    return state


def on_node_complete(engine, state, node):
    """Resolve a previously marked exploration reward at most once."""
    key = "hidden_natural:" + node.node_id
    if engine.counter(state, key) == 1:
        state = engine.set_counter(state, key, 2)
        state = engine.grant(state, "pool:pool_scrap_7", source="hide_and_seek_mark")
    return state


def reward_up_parameters(engine, state, target="rogue_6_gold"):
    """Expose proven modifiers; aggregation/rounding are execution assumptions."""
    return tuple((definition.item_id, float(bb["up"]))
                 for definition in owned_relics(engine, state)
                 for buff in definition.client_buffs
                 if buff["key"] == "up_reward"
                 for bb in (blackboard(buff),)
                 if bb.get("id") == target and bb.get("mask") == "battle")
