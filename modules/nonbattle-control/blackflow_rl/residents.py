"""Resident markers, independently from the underlying exploration nodes.

Spawn radius and movement exclusions are documented by PRTS. Public client
stages establish the combat families, but not their occurrence weights or
general relic/part presence. No automatic reward or respawn is invented here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from .domain import FloorMap, MapNode, NodeType


class ResidentEvidenceRequired(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResidentMarker:
    marker_id: str
    node_id: str


@dataclass(frozen=True, slots=True)
class ResidentContext:
    """Save this per region, and suspend it during a visit to a Black Pond."""
    region_id: str
    stronghold_node_ids: frozenset[str]
    native_empty_node_ids: frozenset[str]
    markers: tuple[ResidentMarker, ...] = ()
    destroyed_node_ids: frozenset[str] = frozenset()
    defeated_stronghold_ids: frozenset[str] = frozenset()
    assumptions: tuple[str, ...] = ()

    @property
    def occupied_node_ids(self) -> frozenset[str]:
        return frozenset(marker.node_id for marker in self.markers)


# Repeatedly enterable destinations and strongholds cannot be occupied.
# LIGHT is intentionally absent: its underlying AP reward remains applicable.
EXCLUDED_DESTINATIONS = frozenset({
    NodeType.START, NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP,
    NodeType.FINAL, NodeType.EVACUATE, NodeType.DOOR, NodeType.STORY,
    NodeType.STORY_HIDDEN, NodeType.BATTLE_SAVAGE,
})


def resident_spawn_candidates(floor_map: FloorMap) -> tuple[str, ...]:
    """Original empty nodes within five actual graph edges of any stronghold."""
    adjacency = floor_map.adjacency()
    strongholds = [n.node_id for n in floor_map.nodes if n.node_type == NodeType.BATTLE_SAVAGE]
    distances = {node_id: 0 for node_id in strongholds}
    queue = deque(strongholds)
    while queue:
        node_id = queue.popleft()
        if distances[node_id] >= 5:
            continue
        for neighbor in adjacency[node_id]:
            if neighbor not in distances:
                distances[neighbor] = distances[node_id]+1
                queue.append(neighbor)
    return tuple(sorted(n.node_id for n in floor_map.nodes
                        if n.node_type == NodeType.EMPTY and n.node_id in distances))


def initialize_residents(
    floor_map: FloorMap, *, remembrance: bool = False,
    observed_positions: tuple[str, ...] | None = None,
    observed_remembrance_cap: int | None = None,
    green_pond: bool = False, rng=None, allow_synthetic: bool = False,
) -> ResidentContext:
    """Initialize once at region entry; this is never a replenishment method.

    Ordinary regions fill up to three candidates. Remembrance samples a cap
    from 0..3. Green Black Ponds contain six markers, independently of the
    ordinary five-edge spawn radius. Unknown placements require observation
    unless the explicitly labelled sampling approximation is enabled.
    """
    native_empty = frozenset(n.node_id for n in floor_map.nodes if n.node_type == NodeType.EMPTY)
    strongholds = frozenset(n.node_id for n in floor_map.nodes if n.node_type == NodeType.BATTLE_SAVAGE)
    assumptions = []
    if green_pond:
        candidates, cap = tuple(sorted(native_empty)), 6
    elif not strongholds:
        candidates, cap = (), 0
    else:
        candidates, cap = resident_spawn_candidates(floor_map), 3
        if remembrance:
            if observed_remembrance_cap is None:
                if not allow_synthetic or rng is None:
                    raise ResidentEvidenceRequired('remembrance resident cap needs observation or explicit sampling')
                cap = rng.randrange(4)
                assumptions.append('remembrance:unverified_uniform_cap_0_to_3')
            else:
                if observed_remembrance_cap not in (0, 1, 2, 3):
                    raise ValueError('remembrance resident cap must be 0..3')
                cap = observed_remembrance_cap
    count = min(cap, len(candidates))
    if observed_positions is None:
        if count and (not allow_synthetic or rng is None):
            raise ResidentEvidenceRequired('initial resident locations need observation or explicit sampling')
        positions = tuple(sorted(rng.sample(candidates, count))) if count else ()
        if count:
            assumptions.append('residents:unverified_uniform_initial_positions')
    else:
        positions = tuple(sorted(observed_positions))
        if len(positions) != count or len(set(positions)) != count or not set(positions).issubset(candidates):
            raise ValueError('observed resident positions violate documented count or spawn constraints')
    return ResidentContext(floor_map.start_node_id, strongholds, native_empty,
                           tuple(ResidentMarker(f'{floor_map.start_node_id}:resident:{index}', node_id)
                                 for index, node_id in enumerate(positions)),
                           assumptions=tuple(assumptions))


def resident_movement_candidates(
    floor_map: FloorMap, context: ResidentContext,
) -> dict[str, tuple[str, ...]]:
    """Each resident moves one connected edge, without changing node types.

    This returns structural possibilities; simultaneous collision handling
    and direction probabilities are not established by the cited rules.
    """
    adjacency = floor_map.adjacency()
    nodes = {n.node_id: n for n in floor_map.nodes}

    def eligible(node: MapNode) -> bool:
        return (node.node_type not in EXCLUDED_DESTINATIONS and
                (not node.repeatable or node.node_type == NodeType.LIGHT) and
                node.node_id not in context.destroyed_node_ids and
                not (node.node_type == NodeType.EMPTY and node.node_id not in context.native_empty_node_ids))

    return {marker.marker_id: tuple(n for n in adjacency[marker.node_id] if eligible(nodes[n]))
            for marker in context.markers}


def move_residents(
    floor_map: FloorMap, context: ResidentContext, *,
    observed_moves: dict[str, str] | None = None,
    rng=None, allow_synthetic: bool = False,
) -> ResidentContext:
    """Advance once per player move, including a multi-edge processed move.

    Explicit sampling uses uniform available neighbors and sequential collision
    avoidance. Both are assumptions, not recovered server probabilities.
    """
    if not context.markers:
        return context
    candidates = resident_movement_candidates(floor_map, context)
    if observed_moves is not None:
        if set(observed_moves) != {m.marker_id for m in context.markers}:
            raise ValueError('observed resident movement must identify every active marker')
        destinations = []
        for marker in context.markers:
            destination = observed_moves[marker.marker_id]
            if destination not in candidates[marker.marker_id] and not (
                    destination == marker.node_id and not candidates[marker.marker_id]):
                raise ValueError('resident move must follow one eligible edge')
            destinations.append(destination)
        if len(destinations) != len(set(destinations)):
            raise ResidentEvidenceRequired('overlapping resident markers require a collision model')
        return replace(context, markers=tuple(replace(m, node_id=observed_moves[m.marker_id]) for m in context.markers))
    if not allow_synthetic or rng is None:
        raise ResidentEvidenceRequired('resident directions need observation or explicit sampling')
    occupied = set(context.occupied_node_ids)
    markers = []
    for marker in context.markers:
        occupied.remove(marker.node_id)
        choices = tuple(n for n in candidates[marker.marker_id] if n not in occupied)
        destination = rng.choice(choices) if choices else marker.node_id
        occupied.add(destination)
        markers.append(replace(marker, node_id=destination))
    assumption = 'residents:unverified_uniform_directions_sequential_collision_avoidance'
    return replace(context, markers=tuple(markers),
                   assumptions=tuple(dict.fromkeys(context.assumptions+(assumption,))))


def walking_path_blocked(context: ResidentContext, path: tuple[str, ...]) -> bool:
    """A player may enter a resident node, but may not walk through it."""
    return bool(context.occupied_node_ids.intersection(path[1:-1]))


def resolve_resident_victory(
    floor_map: FloorMap, context: ResidentContext, node_id: str,
) -> tuple[FloorMap, ResidentContext]:
    """Remove a defeated marker, or clear every marker after one stronghold.

    The occupied node is destroyed into EMPTY; its original event/reward is
    lost. No second transaction, growth, fresh resident or loot is generated.
    """
    if node_id in context.stronghold_node_ids and node_id not in context.defeated_stronghold_ids:
        context = replace(context, markers=(), defeated_stronghold_ids=context.defeated_stronghold_ids | {node_id})
    elif node_id in context.occupied_node_ids:
        context = replace(context, markers=tuple(m for m in context.markers if m.node_id != node_id))
    else:
        raise ValueError('no undefeated resident or stronghold at this node')
    context = replace(context, destroyed_node_ids=context.destroyed_node_ids | {node_id})
    nodes = tuple(replace(n, node_type=NodeType.EMPTY, options=(), event_name=None,
                          auto_effect=type(n.auto_effect)(), repeatable=False, stage_id=None)
                  if n.node_id == node_id else n for n in floor_map.nodes)
    return replace(floor_map, nodes=nodes, fingerprint=''), context


@dataclass(frozen=True, slots=True)
class ResidentBattle:
    stage_id: str
    name: str
    gold: int
    command_exp: int
    stronghold: bool
    relic_pool: str | None = None
    # Pool membership is not evidence for reward presence, count or probability.
    relic_reward_count: int | None = None
    part_reward_count: int | None = None


RESIDENT_BATTLES = {
    **{f'ro6_t_8{suffix}': ResidentBattle(f'ro6_t_8{suffix}', '强买强卖', 3, 15, False)
       for suffix in ('', '_b', '_c')},
    **{f'ro6_t_9{suffix}': ResidentBattle(f'ro6_t_9{suffix}', '进退趋同', 5, 18, False)
       for suffix in ('', '_b', '_c')},
    'ro6_t_10': ResidentBattle('ro6_t_10', '枯枝', 5, 25, True, 'rogue_6:居民据点'),
    'ro6_t_11': ResidentBattle('ro6_t_11', '败叶', 5, 30, True, 'rogue_6:居民据点'),
}


def select_resident_battle(*, stronghold: bool, observed_stage_id: str | None = None,
                           rng=None, allow_synthetic: bool = False) -> ResidentBattle:
    if observed_stage_id is not None:
        battle = RESIDENT_BATTLES[observed_stage_id]
        if battle.stronghold != stronghold:
            raise ValueError('stage does not belong to this resident encounter family')
        return battle
    if not allow_synthetic or rng is None:
        raise ResidentEvidenceRequired('resident stage occurrence and floor gates require observation')
    return rng.choice(tuple(b for b in RESIDENT_BATTLES.values() if b.stronghold == stronghold))
