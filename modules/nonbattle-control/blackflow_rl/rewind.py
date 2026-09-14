"""One-use Remembrance redirect to a newly generated fourth region.

PRTS No.216 confirms a different fourth-region map, treated as a new region.
The ordinary exit pipeline therefore remains responsible for AP carry/hope,
floor taxes, temporary-part expiry, entry rewards and progression. This module
only inserts the redirected destination and pays the two item-cost options.
"""
from dataclasses import replace

from .domain import FloorMap, GameState


REMEMBRANCE = 'rogue_6_relic_artifact_5'
PRIVATE_KEY = 'rogue_6_relic_cargo_13'
SPENT_KEY = 'remembrance:spent'


def has_pending_remembrance(state: GameState) -> bool:
    owned = REMEMBRANCE in state.inventory or any(x.item_id == REMEMBRANCE for x in state.item_instances)
    return owned and not dict(state.event_counters).get(SPENT_KEY, 0)


def pay_remembrance_item_cost(engine, state: GameState, operation: str) -> GameState:
    """Consume actual held instances, then acquire the named relic once."""
    if REMEMBRANCE in state.inventory:
        raise ValueError('cannot pay for a unique relic already held')
    if operation == 'remembrance_parts':
        candidates = sorted(x.instance_id for x in state.item_instances if x.category == 'MOVE')
        if len(candidates) < 2:
            raise ValueError('remembrance requires two held processed items')
        rng, state = engine.roll(state, 'remembrance:random_two_move')
        state = engine.entry(state, 'model_assumption', source='remembrance:uniform_held_move_instances')
        for instance_id in rng.sample(candidates, 2):
            state = engine.remove(state, instance_id, 'consume', 'remembrance_parts')
    elif operation == 'remembrance_key':
        key = next((x for x in state.item_instances if x.item_id == PRIVATE_KEY), None)
        if key is None:
            raise ValueError('remembrance requires the source private key')
        state = engine.remove(state, key.instance_id, 'consume', 'remembrance_key')
    else:
        raise ValueError('unknown remembrance payment operation')
    return engine.acquire(state, REMEMBRANCE, 'event:愈创之心')


def namespace_rewind_map(floor_map: FloorMap, namespace: str = 'F4R1') -> FloorMap:
    """Distinct node identities prevent old shop stock/counts from leaking in."""
    node_ids = {n.node_id: f'{namespace}_N{n.index:02d}' for n in floor_map.nodes}
    return replace(floor_map, nodes=tuple(replace(n, node_id=node_ids[n.node_id]) for n in floor_map.nodes),
                   edges=tuple((node_ids[a], node_ids[b]) for a, b in floor_map.edges),
                   start_node_id=node_ids[floor_map.start_node_id],
                   exit_node_ids=tuple(node_ids[n] for n in floor_map.exit_node_ids), fingerprint='')


def prepare_remembrance_redirect(state: GameState, replacement_map: FloorMap) -> GameState:
    """Insert an observed/generated new IV map; caller then advances normally.

    No ordinary-node distributions are defined here. Pass a map from the
    versioned generator with its explicit sampling profile, or an observation.
    Global visited-event exclusions are intentionally retained. Residents in
    the destination use the special remembered-region 0..3 initial cap.
    """
    if not has_pending_remembrance(state):
        raise ValueError('no unspent remembrance')
    if getattr(state, 'portal_context', None) is not None:
        raise ValueError('Black Pond entrance/return does not trigger remembrance')
    if state.floor != 4 or replacement_map.floor != 4:
        raise ValueError('documented remembrance redirect applies to region IV')
    if replacement_map is state.floor_map or replacement_map.fingerprint == state.floor_map.fingerprint:
        raise ValueError('remembrance must lead to a different fourth-region map')
    replacement_map = namespace_rewind_map(replacement_map)
    if set(n.node_id for floor in state.maps for n in floor.nodes) & set(n.node_id for n in replacement_map.nodes):
        raise ValueError('remembrance destination node ids collide')
    maps = state.maps[:state.floor_index+1] + (replacement_map,) + state.maps[state.floor_index+1:]
    counters = dict(state.event_counters)
    counters[SPENT_KEY] = 1
    counters['remembrance:destination_index'] = state.floor_index+1
    return replace(state, maps=maps, event_counters=tuple(sorted(counters.items())))
