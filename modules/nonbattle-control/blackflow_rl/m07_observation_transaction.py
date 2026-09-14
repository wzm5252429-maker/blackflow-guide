"""Unwired prototype of the opt-in M07 observation transaction boundary.

The absent-observation branch delegates to the existing simulator unchanged.
An acknowledged real use instead locks a session after physical consumption.
This prototype deliberately has no normal-arrival or region-exit executor:
both require a future reviewed dispatcher, not a second simulator move.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .domain import ActionKind, BATTLE_TYPES, NodeType
from .m07_surprise import (M07, SurpriseContext, SurpriseMenuObservation,
                          open_observed_surprise)


class ObservationPending(ValueError):
    """An actual use is committed; do not retry movement or simulate a fallback."""


def _source(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must identify an actual observation')


@dataclass(frozen=True, slots=True)
class ObservedM07Use:
    observation_id: str
    instance_id: str
    movement_count_before: int
    source: str


@dataclass(frozen=True, slots=True)
class ObservedM07Destination:
    observation_id: str
    use_observation_id: str
    kind: str  # ordinary or surprise; supplied from the actual UI, not sampled
    source: str
    node_id: str | None = None
    node_type: NodeType | None = None
    menu: SurpriseMenuObservation | None = None


@dataclass(frozen=True, slots=True)
class OrdinaryArrivalRequest:
    request_id: str
    node_id: str
    observed_node_type: NodeType
    movement_already_consumed: bool = True
    source: str = ''


@dataclass(frozen=True, slots=True)
class PendingM07Use:
    use: ObservedM07Use
    before_move: Any
    after_consumption: Any
    phase: str = 'AWAITING_DESTINATION'
    destination: ObservedM07Destination | None = None
    ordinary_arrival: OrdinaryArrivalRequest | None = None


@dataclass(frozen=True, slots=True)
class M07ObservationSession:
    run_id: str
    state: Any
    pending: PendingM07Use | None = None
    observation_ids: frozenset[str] = frozenset()
    surprise: SurpriseContext = SurpriseContext()


@dataclass(frozen=True, slots=True)
class SessionActionResult:
    session: M07ObservationSession
    # Normal simulation returns its exact Transition. A pause has no extra
    # synthetic transition, objective reward, chase, or step increment.
    transition: Any | None = None


def session_legal_actions(simulator, session):
    return () if session.pending is not None else simulator.legal_actions(session.state)


def session_action(simulator, session, action_id, *, observed_use=None):
    if session.pending is not None:
        raise ObservationPending('the committed M07 use still needs observed destination settlement')
    if observed_use is None:
        # Do not decode again, branch on inventory, inspect unseen nodes,
        # construct a receipt, or touch RNG on the default simulation path.
        transition = simulator.transition(session.state, action_id)
        return SessionActionResult(replace(session, state=transition.next_state), transition)
    if not isinstance(observed_use, ObservedM07Use):
        raise ValueError('a typed actual-use acknowledgement is required')
    _source(session.run_id, 'run_id')
    _source(observed_use.observation_id, 'observation_id')
    _source(observed_use.source, 'source')
    if observed_use.observation_id in session.observation_ids:
        raise ValueError('actual-use observation was already committed')
    state = session.state
    if (type(observed_use.movement_count_before) is not int
            or observed_use.movement_count_before != state.movement_count
            or state.portal_context is not None or state.floor not in (1, 2, 3, 4)
            or state.pending_node_id is not None or state.awaiting_exit or state.terminal):
        raise ValueError('unsupported or stale ordinary-area M07 use acknowledgement')
    action = simulator.decode_action(state, action_id)
    item = next((x for x in state.item_instances
                 if x.instance_id == state.equipped_instance_id), None)
    if (action.kind != ActionKind.MOVE or action.movement_cost != 0
            or item is None or item.item_id != M07 or item.category != 'MOVE'
            or item.uses_remaining != 1 or item.instance_id != observed_use.instance_id
            or action.equipment_instance_id != item.instance_id):
        raise ValueError('acknowledgement must match this legal equipped M07 activation')
    # Only the already observed physical use is recorded. The destination is
    # deliberately still unknown; do not call roll, on_move, or enter here.
    consumed = simulator.economy.remove(state, item.instance_id, 'consume', 'movement')
    consumed = replace(consumed, movement_count=state.movement_count + 1)
    pending = PendingM07Use(observed_use, state, consumed)
    return SessionActionResult(replace(session, state=consumed, pending=pending,
        observation_ids=session.observation_ids | {observed_use.observation_id}))


def observe_m07_destination(session, observation):
    pending = session.pending
    if pending is None:
        raise ValueError('observe only the destination of an actually committed M07 use')
    if not isinstance(observation, ObservedM07Destination):
        raise ValueError('a typed destination observation is required')
    if pending.destination is not None:
        if observation == pending.destination:
            return session  # Same callback retry: no duplicate command or grant.
        raise ValueError('a committed destination cannot be replaced or rerolled')
    _source(observation.observation_id, 'observation_id')
    _source(observation.source, 'source')
    if (observation.use_observation_id != pending.use.observation_id
            or observation.observation_id in session.observation_ids):
        raise ValueError('destination observation belongs to another use or was replayed')
    if session.state != pending.after_consumption:
        raise ValueError('the pending physical-consumption state changed before destination observation')
    surprise = session.surprise
    arrival = None
    ids = session.observation_ids | {observation.observation_id}
    if observation.kind == 'surprise':
        if (observation.node_id is not None or observation.node_type is not None
                or observation.menu is None
                or observation.menu.observation_id in session.observation_ids):
            raise ValueError('the hidden event needs its actual menu, not an ordinary map node')
        surprise = open_observed_surprise(surprise,
            before_move=pending.before_move, after_consumption=pending.after_consumption,
            instance_id=pending.use.instance_id, run_id=session.run_id, menu=observation.menu)
        ids |= {observation.menu.observation_id}
        phase = 'SURPRISE_OBSERVED'
    elif observation.kind == 'ordinary':
        if (observation.menu is not None or not isinstance(observation.node_type, NodeType)
                or observation.node_type in BATTLE_TYPES
                or observation.node_id not in {n.node_id for n in session.state.floor_map.nodes}):
            raise ValueError('ordinary destination needs an observed nonbattle map coordinate and type')
        # Do not infer the observed type from the simulator's hidden payload.
        # A future resolver must reconcile this observation before entering it.
        arrival = OrdinaryArrivalRequest(pending.use.observation_id + ':ordinary-arrival',
            observation.node_id, observation.node_type, source=observation.source)
        phase = 'ORDINARY_ARRIVAL_OBSERVED'
    else:
        raise ValueError('unknown destination kind; remain paused without a fallback draw')
    return replace(session, pending=replace(pending, phase=phase,
        destination=observation, ordinary_arrival=arrival), surprise=surprise,
        observation_ids=ids)
