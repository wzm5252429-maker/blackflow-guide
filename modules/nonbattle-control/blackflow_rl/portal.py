"""Black Pond layouts and isolated AP, with no invented graph expansion.

The nine graphs are pinned community-extracted topologies. Client data proves
nineteen zone identities and their AP grants. Unknown correspondence, contents
and sampling weights require an explicitly synthetic call; observed node types
can instead be supplied. This module never claims to simulate combat rosters.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Mapping

from .domain import FloorMap, GameState, MapNode, NodeType, PortalContext, ResourceDelta


SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / 'data/evidence/rogue6_portal_layouts_v1.json'
BLUE_NODE_TYPES = (NodeType.INCIDENT, NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE, NodeType.BATTLE_SHOP, NodeType.WISH)


class PortalEvidenceRequired(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PortalLayout:
    floor_map: FloorMap
    zone_id: str
    variation_id: int
    action_points: int
    resident_node_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def load_portal_evidence() -> dict:
    data = json.loads(SNAPSHOT_PATH.read_text(encoding='utf8'))
    if len(data['templates']) != 9 or len(data['zone_action_points']) != 19:
        raise ValueError('portal evidence cardinality changed')
    return data


def portal_variation(zone_id: str) -> int:
    if zone_id not in load_portal_evidence()['zone_action_points']:
        raise ValueError(f'unknown portal zone: {zone_id}')
    return int(zone_id.removeprefix('zone_portal_normal_').split('_', 1)[0])


def portal_ap(zone_id: str) -> int:
    return load_portal_evidence()['zone_action_points'][zone_id]


def compatible_portal_templates(zone_id: str) -> tuple[str, ...]:
    """Compatibility constraints, not a proved zone-to-template mapping."""
    variation = portal_variation(zone_id)
    candidates = []
    for template in load_portal_evidence()['templates']:
        if variation in template.get('utopiaPortals', (template.get('utopiaPortal'),)):
            if template.get('action', 3) == portal_ap(zone_id):
                candidates.append(template['id'])
    return tuple(candidates)


def build_portal_layout(
    zone_id: str, *, outer_floor: int, seed: int, namespace: str,
    template_id: str | None = None,
    observed_node_types: Mapping[tuple[int, int], NodeType] | None = None,
    observed_resident_slots: tuple[tuple[int, int], ...] | None = None,
    allow_synthetic: bool = False,
) -> PortalLayout:
    if outer_floor not in (3, 4, 5):
        raise ValueError('ordinary portals only occur in floors III, IV, V')
    if not namespace:
        raise ValueError('portal node namespace must be unique and non-empty')
    variation = portal_variation(zone_id)
    candidates = compatible_portal_templates(zone_id)
    assumptions = []
    rng = random.Random(seed)
    if template_id is None:
        if len(candidates) != 1 and not allow_synthetic:
            raise PortalEvidenceRequired('red-zone topology correspondence is unobserved')
        template_id = rng.choice(candidates)
        if len(candidates) != 1:
            assumptions.append('synthetic_uniform_red_topology_with_matching_client_ap')
    if template_id not in candidates:
        raise ValueError('topology incompatible with fog family or client AP grant')
    template = next(t for t in load_portal_evidence()['templates'] if t['id'] == template_id)
    points = sorted((y, x) for x, y in template['occupiedSlots'])
    start = tuple(reversed(template['startSlot']))
    edges = tuple((tuple(reversed(left)), tuple(reversed(right))) for left, right in template['edges'])
    neighbors = {point: [] for point in points}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    distances = {start: 0}
    queue = deque((start,))
    while queue:
        point = queue.popleft()
        for target in neighbors[point]:
            if target not in distances:
                distances[target] = distances[point]+1
                queue.append(target)
    if len(distances) != len(points):
        raise ValueError('disconnected source topology')
    content = {start: NodeType.START}
    content.update({tuple(reversed(n['slot'])): NodeType.BATTLE_SAVAGE for n in template.get('fixedNodes', ())})
    if observed_node_types is not None:
        if set(observed_node_types) - set(points):
            raise ValueError('observed nodes cannot expand the source topology')
        for slot, kind in observed_node_types.items():
            if slot in content and content[slot] != NodeType(kind):
                raise ValueError('observation conflicts with a fixed source slot')
            content[slot] = NodeType(kind)
    missing = set(points)-set(content)
    if missing:
        if not allow_synthetic:
            raise PortalEvidenceRequired('all unknown source slots require observation')
        if variation == 8:
            pool = (NodeType.DUEL,)
        elif variation == 9:
            pool = (NodeType.SACRIFICE,)
        elif variation == 6:
            pool = (NodeType.EMPTY,)
        elif variation == 5:
            pool = BLUE_NODE_TYPES
            assumptions.append('synthetic_uniform_blue_five_type_pool')
        elif variation == 7:
            pool = (rng.choice((NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE)),)
            assumptions.append('synthetic_uniform_gold_battle_type_shared_by_all_slots')
        else:
            pool = (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE)
            assumptions.append('synthetic_uniform_red_battle_types')
        for point in sorted(missing):
            content[point] = rng.choice(pool)
        assumptions.append('community_construction_family_fills_unobserved_slots')
    ids = {point: f'{namespace}_N{i:02d}' for i, point in enumerate(points)}
    nodes = tuple(MapNode(ids[p], i, p[0], p[1], content[p], distances[p],
                          repeatable=content[p] in (NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP))
                  for i, p in enumerate(points))
    residents = ()
    if variation == 6:
        if observed_resident_slots is None:
            if not allow_synthetic:
                raise PortalEvidenceRequired('six resident positions require observation')
            resident_slots = rng.sample([p for p in points if content[p] == NodeType.EMPTY], 6)
            assumptions.append('synthetic_uniform_six_resident_initial_positions')
        else:
            resident_slots = observed_resident_slots
            if len(set(resident_slots)) != 6 or any(p not in ids or p == start or content[p] != NodeType.EMPTY for p in resident_slots):
                raise ValueError('resident observation requires six distinct eligible slots')
        residents = tuple(sorted(ids[p] for p in resident_slots))
    floor_map = FloorMap(outer_floor, template['gridShape'][0], template['gridShape'][1], nodes,
                         tuple((ids[a], ids[b]) for a, b in edges), ids[start], (), seed)
    return PortalLayout(floor_map, zone_id, variation, portal_ap(zone_id), residents, tuple(assumptions))


def portal_entry_options(state: GameState, rng: random.Random) -> tuple[str, ...]:
    """At most three held MOVE instances offered for whole-item consumption.

    Uniform subset selection is a synthetic prior. Remaining uses do not turn
    one instance into several entrance tickets, and non-MOVE parts cannot pay.
    """
    candidates = sorted(x.instance_id for x in state.item_instances if x.category == 'MOVE')
    return tuple(rng.sample(candidates, min(3, len(candidates))))


def enter_portal_state(state: GameState, layout: PortalLayout) -> tuple[GameState, PortalContext]:
    """Caller consumes the chosen MOVE and finishes the entrance node first."""
    if state.pending_node_id is not None:
        raise ValueError('resolve the entrance interaction before suspending it')
    if state.floor != layout.floor_map.floor:
        raise ValueError('portal keeps the ordinary floor number')
    if set(n.node_id for n in state.floor_map.nodes) & set(n.node_id for n in layout.floor_map.nodes):
        raise ValueError('portal namespace collides with the outer map')
    context = PortalContext(layout.zone_id, layout.variation_id, state.floor_map, state.current_node_id,
                            state.resources.action_points, state.completed, state.revealed,
                            state.seen_event_names, layout.resident_node_ids,outer_region_state=state.region_state,
                            outer_resident_context=state.resident_context, outer_fate_context=state.fate_context)
    maps = list(state.maps)
    maps[state.floor_index] = layout.floor_map
    updated = replace(state, maps=tuple(maps), current_node_id=layout.floor_map.start_node_id,
                      resources=replace(state.resources, action_points=layout.action_points),
                      completed=frozenset({layout.floor_map.start_node_id}),
                      revealed=frozenset({layout.floor_map.start_node_id}), awaiting_exit=False,
                      region_state=None, resident_context=None, fate_context=None)
    return updated, context


def return_portal_state(state: GameState, context: PortalContext) -> GameState:
    """Restore suspended map/AP without floor tax, chase, expiry or hope gain."""
    if state.pending_node_id is not None or state.resources.parts > state.parts_capacity:
        raise ValueError('resolve interaction and inventory overflow before portal return')
    maps = list(state.maps)
    maps[state.floor_index] = context.outer_floor_map
    return replace(state, maps=tuple(maps), current_node_id=context.outer_node_id,
                   resources=replace(state.resources, action_points=context.outer_action_points),
                   completed=context.outer_completed, revealed=context.outer_revealed,
                   awaiting_exit=False,region_state=context.outer_region_state,resident_context=context.outer_resident_context,
                   fate_context=context.outer_fate_context)


def portal_completion_reward(context: PortalContext, node: MapNode) -> tuple[PortalContext, ResourceDelta]:
    """Apply a completion bounty at most once per actual content node."""
    if node.node_id in context.rewarded_node_ids or node.node_type in (NodeType.START, NodeType.EMPTY):
        return context, ResourceDelta()
    gold = 10 if context.variation_id == 7 else 5 if context.variation_id == 9 and node.node_type == NodeType.SACRIFICE else 0
    return replace(context, rewarded_node_ids=context.rewarded_node_ids | {node.node_id}), ResourceDelta(gold=gold)


def blue_refresh_unvisited(floor_map: FloorMap, *, visited: frozenset[str], current_node_id: str,
                           seed: int, allow_synthetic: bool = False) -> FloorMap:
    if not allow_synthetic:
        raise PortalEvidenceRequired('blue node reroll weights are unobserved')
    updated = []
    for node in floor_map.nodes:
        if node.node_id in visited or node.node_id == current_node_id or node.node_type == NodeType.START:
            updated.append(node)
            continue
        digest = sha256(f'{seed}:{node.node_id}:blue'.encode()).digest()
        kind = random.Random(int.from_bytes(digest[:8], 'big')).choice(BLUE_NODE_TYPES)
        updated.append(replace(node, node_type=kind, event_name=None, options=(), auto_effect=ResourceDelta(),
                               repeatable=kind == NodeType.BATTLE_SHOP, requires_observation=False))
    return replace(floor_map, nodes=tuple(updated), fingerprint='')


def record_spiral_victory(context: PortalContext, operator_ids: tuple[str, ...]) -> PortalContext:
    if context.variation_id != 8:
        return context
    if context.exhausted_operator_ids.intersection(operator_ids):
        raise ValueError('an exhausted spiral operator cannot formally fight again')
    return replace(context, exhausted_operator_ids=context.exhausted_operator_ids | set(operator_ids))
