"""Observed M07 hidden-event transactions, intentionally not runtime-wired.

No trigger rate, random successor or reward pool is sampled here.  Integration
must pause a real M07 move after its physical consumption and before ordinary
destination dispatch.  A fully resolved ordinary move cannot be retrofitted.
The immutable context and emitted command must be persisted atomically by a
future bridge; discarding the returned context is not a supported replay API.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256


M07 = 'rogue_6_scrap_M_07'
PRIVATE_KEY = 'rogue_6_relic_cargo_13'
ENTRY_SCENE = 'scene_ro6_surprise_enter'
END_SCENE = 'scene_ro6_surprise_8'
SCENES = frozenset({ENTRY_SCENE} | {f'scene_ro6_surprise_{i}' for i in range(1, 12)})
ENTRY_CHOICES = frozenset({'choice_ro6_surprise_1', 'choice_ro6_surprise_12'})
RANDOM_CHOICES = frozenset(f'choice_ro6_surprise_{i}' for i in (5, 6, 9, 11))
PART_CHOICES = frozenset(f'choice_ro6_surprise_{i}' for i in (4, 8))
CHOICES = frozenset(f'choice_ro6_surprise_{i}' for i in range(1, 13))
NEXT_SCENE = {f'choice_ro6_surprise_{i}': f'scene_ro6_surprise_{n}'
              for i, n in ((1, 1), (12, 1), (2, 5), (3, 9), (4, 11),
                           (7, 8), (8, 8), (10, 8))}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must identify a real observation')
    return value


@dataclass(frozen=True, slots=True)
class SurpriseMenuObservation:
    observation_id: str
    scene_id: str
    choice_ids: tuple[str, ...]
    source: str


@dataclass(frozen=True, slots=True)
class SurpriseReward:
    """Exactly one command; PRIVATE_KEY acquire already pays its own 99 gold."""
    command_id: str
    choice_id: str
    item_id: str | None = None
    gold: int = 0
    hope: int = 0
    shield: int = 0
    observation_id: str = ''
    source: str = ''


@dataclass(frozen=True, slots=True)
class SurpriseAudit:
    observation_id: str
    source: str
    kind: str
    detail: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurpriseExitRequest:
    """A request, not an advance or an ordinary FINAL/EVAC reward entitlement."""
    request_id: str
    from_floor: int
    to_floor: int
    movement_count: int
    observed_ap_rule: None = None  # No AP conversion/carry rule was established.


@dataclass(frozen=True, slots=True)
class SurpriseVisit:
    visit_id: str
    run_id: str
    floor: int
    origin_node_id: str
    consumed_instance_id: str
    movement_count: int
    menu: SurpriseMenuObservation | None = None
    claimed_choice_ids: frozenset[str] = frozenset()
    awaiting_scene_id: str | None = None
    awaiting_random_successor: bool = False
    awaiting_part_choice_id: str | None = None
    entry_reward_claimed: bool = False
    exit_request: SurpriseExitRequest | None = None


@dataclass(frozen=True, slots=True)
class SurpriseContext:
    active: SurpriseVisit | None = None
    used_move_ids: frozenset[str] = frozenset()
    observation_ids: frozenset[str] = frozenset()
    completed_visit_ids: frozenset[str] = frozenset()
    audit: tuple[SurpriseAudit, ...] = ()


@dataclass(frozen=True, slots=True)
class SurpriseTransition:
    context: SurpriseContext
    reward: SurpriseReward | None = None
    exit_request: SurpriseExitRequest | None = None


def _active(context):
    if context.active is None or context.active.exit_request is not None:
        raise ValueError('a live unresolved observed surprise event is required')
    return context.active


def _validate_consumption(before, after, instance_id):
    """Check the exact no-destination pause using existing GameState snapshots.

    No movement hook, vehicle-replacement hook, reveal, resident movement,
    destination reward or region change is authorized by this helper. Future
    integration must settle applicable hooks separately once their semantics
    for this special destination are observed. The used M07 cannot survive.
    """
    if (before.terminal or before.pending_node_id is not None or before.awaiting_exit
            or before.portal_context is not None or before.floor not in (1, 2, 3, 4)):
        raise ValueError('only a live ordinary-area I-IV navigation use is supported')
    item = next((x for x in before.item_instances if x.instance_id == instance_id), None)
    if (item is None or item.item_id != M07 or item.category != 'MOVE'
            or item.uses_remaining != 1 or before.equipped_instance_id != instance_id):
        raise ValueError('the actual equipped one-charge M07 instance is required')
    if len(after.ledger) != len(before.ledger) + 1 or after.ledger[:-1] != before.ledger:
        raise ValueError('expected exactly the physical movement-consumption receipt')
    entry = after.ledger[-1]
    if (entry.step != before.step_count + 1 or entry.floor != before.floor
            or entry.node_id != before.current_node_id or entry.operation != 'consume'
            or entry.item_id != M07 or entry.instance_id != instance_id
            or entry.category != 'MOVE' or entry.quantity != 1
            or entry.gold_delta != 0 or entry.source != 'movement'):
        raise ValueError('the receipt does not prove this actual M07 movement consumption')
    expected = replace(before,
        item_instances=tuple(x for x in before.item_instances if x.instance_id != instance_id),
        equipped_instance_id=None, movement_count=before.movement_count + 1,
        resources=replace(before.resources, parts=before.resources.parts - 1),
        ledger=after.ledger)
    if after != expected:
        raise ValueError('pause must precede destination/reward/extra-movement settlement')


def _validated_menu(menu):
    if not isinstance(menu, SurpriseMenuObservation):
        raise ValueError('an explicit observed menu is required')
    _text(menu.observation_id, 'observation_id')
    _text(menu.source, 'source')
    if menu.scene_id not in SCENES or not isinstance(menu.choice_ids, tuple):
        raise ValueError('unsupported hidden-event scene or menu representation')
    if len(menu.choice_ids) != len(set(menu.choice_ids)) or any(x not in CHOICES for x in menu.choice_ids):
        raise ValueError('observed choices must be distinct supported client IDs')
    if menu.scene_id == END_SCENE:
        if menu.choice_ids:
            raise ValueError('the observed ending scene cannot pay another choice')
    elif not menu.choice_ids:
        raise ValueError('an empty nonterminal scene is unresolved, not completion')
    return menu


def open_observed_surprise(context, *, before_move, after_consumption,
                           instance_id, run_id, menu):
    """Accept only a witnessed special destination after one real paid use.

    after_consumption must be the physical-consumption pause, not the output
    of today's complete simulator.transition. This function changes no game
    state, charges, AP or loot and does not authorize re-running a move.
    """
    if context.active is not None:
        raise ValueError('finish the active hidden event first')
    _text(run_id, 'run_id')
    _validate_consumption(before_move, after_consumption, instance_id)
    menu = _validated_menu(menu)
    if menu.scene_id != ENTRY_SCENE or len(menu.choice_ids) != 1 or menu.choice_ids[0] not in ENTRY_CHOICES:
        raise ValueError('observe exactly one of the mutually exclusive entry rewards')
    if menu.choice_ids[0] == 'choice_ro6_surprise_1' and PRIVATE_KEY in before_move.inventory:
        raise ValueError('a duplicate private-key entry requires separate observed-rule resolution')
    visit_id = sha256(repr((run_id, after_consumption.movement_count, instance_id)).encode()).hexdigest()
    if visit_id in context.used_move_ids or menu.observation_id in context.observation_ids:
        raise ValueError('this move or observation has already opened an event')
    visit = SurpriseVisit(visit_id, run_id, before_move.floor, before_move.current_node_id,
        instance_id, after_consumption.movement_count, menu=menu)
    return replace(context, active=visit, used_move_ids=context.used_move_ids | {visit_id},
        observation_ids=context.observation_ids | {menu.observation_id},
        audit=context.audit + (SurpriseAudit(menu.observation_id, menu.source,
            'entry_after_consumption', (visit_id, instance_id, menu.scene_id) + menu.choice_ids),))


def observe_surprise_menu(context, menu):
    """Follow actual screen output; NEXT_PROB does not select a successor here.

    Client choice-to-nextScene IDs are known, but scene-to-menu membership and
    NEXT_PROB tables are incomplete. We do not manufacture their bindings.
    """
    visit = _active(context)
    menu = _validated_menu(menu)
    if visit.menu is not None or visit.awaiting_part_choice_id:
        raise ValueError('resolve the current displayed choice/reward before another observation')
    if menu.observation_id in context.observation_ids:
        raise ValueError('menu observation was already consumed')
    if menu.scene_id == ENTRY_SCENE or any(x in ENTRY_CHOICES for x in menu.choice_ids):
        raise ValueError('the mutually exclusive entry reward cannot recur')
    if not visit.entry_reward_claimed:
        raise ValueError('the observed entry reward must first be settled')
    if not visit.awaiting_random_successor and menu.scene_id != visit.awaiting_scene_id:
        raise ValueError('observed scene conflicts with the deterministic client nextSceneId')
    if any(x in visit.claimed_choice_ids for x in menu.choice_ids):
        raise ValueError('repeated choice rewards/loops are not established for this event')
    return replace(context, active=replace(visit, menu=menu,
        awaiting_scene_id=None, awaiting_random_successor=False),
        observation_ids=context.observation_ids | {menu.observation_id},
        audit=context.audit + (SurpriseAudit(menu.observation_id, menu.source,
            'menu', (menu.scene_id,) + menu.choice_ids),))


def choose_observed_surprise(context, choice_id):
    visit = _active(context)
    if (visit.menu is None or choice_id not in visit.menu.choice_ids
            or choice_id in visit.claimed_choice_ids):
        raise ValueError('only a currently displayed unclaimed choice may execute')
    if choice_id in ENTRY_CHOICES and visit.entry_reward_claimed:
        raise ValueError('the event entry reward was already paid')
    reward = None
    command_id = visit.visit_id + ':' + choice_id
    if choice_id == 'choice_ro6_surprise_1':
        reward = SurpriseReward(command_id, choice_id, item_id=PRIVATE_KEY)
    elif choice_id == 'choice_ro6_surprise_12':
        reward = SurpriseReward(command_id, choice_id, gold=99)
    elif choice_id in ('choice_ro6_surprise_2', 'choice_ro6_surprise_10'):
        reward = SurpriseReward(command_id, choice_id, hope=20)
    elif choice_id in ('choice_ro6_surprise_3', 'choice_ro6_surprise_7'):
        reward = SurpriseReward(command_id, choice_id, shield=15)
    if reward is not None:
        reward = replace(reward, observation_id=visit.menu.observation_id, source=visit.menu.source)
    visit = replace(visit, menu=None,
        claimed_choice_ids=visit.claimed_choice_ids | {choice_id},
        awaiting_scene_id=NEXT_SCENE.get(choice_id),
        awaiting_random_successor=choice_id in RANDOM_CHOICES,
        awaiting_part_choice_id=choice_id if choice_id in PART_CHOICES else None,
        entry_reward_claimed=visit.entry_reward_claimed or choice_id in ENTRY_CHOICES)
    return SurpriseTransition(replace(context, active=visit), reward=reward)


def observe_surprise_part(context, *, observation_id, item_id, source):
    """One actual MOVE outcome, never a pool draw or a candidate reroll."""
    visit = _active(context)
    _text(observation_id, 'observation_id')
    _text(source, 'source')
    if not visit.awaiting_part_choice_id or observation_id in context.observation_ids:
        raise ValueError('a unique observation of the pending part reward is required')
    from .catalog import load_catalog
    definition = load_catalog().items.get(item_id)
    if definition is None or definition.category != 'MOVE':
        raise ValueError('the observed reward must name an actual client MOVE item')
    choice_id = visit.awaiting_part_choice_id
    return SurpriseTransition(replace(context,
        active=replace(visit, awaiting_part_choice_id=None),
        observation_ids=context.observation_ids | {observation_id},
        audit=context.audit + (SurpriseAudit(observation_id, source,
            'part_reward', (choice_id, item_id)),)),
        reward=SurpriseReward(visit.visit_id + ':' + choice_id, choice_id, item_id=item_id,
            observation_id=observation_id, source=source))


def request_surprise_exit(context):
    visit = _active(context)
    if (visit.menu is None or visit.menu.scene_id != END_SCENE
            or visit.menu.choice_ids or not visit.entry_reward_claimed):
        raise ValueError('only an actually observed terminal scene requests a region exit')
    request = SurpriseExitRequest(visit.visit_id + ':exit', visit.floor,
                                  visit.floor + 1, visit.movement_count)
    return SurpriseTransition(replace(context, active=replace(visit, exit_request=request)),
                              exit_request=request)


def confirm_surprise_exit(context, *, request_id, actual_state, observation_id, source):
    """Acknowledge an observed real region transition without executing it.

    The future bridge must resolve the unknown AP/utopia rules from actual
    observations before ordinary region callbacks. This is not portal return
    and gives no extra vehicle or max-HP reward for an ordinary safe exit.
    """
    _text(observation_id, 'observation_id')
    _text(source, 'source')
    visit = context.active
    if (visit is None or visit.exit_request is None
            or visit.exit_request.request_id != request_id
            or observation_id in context.observation_ids):
        raise ValueError('the single pending real region-exit request is required')
    if (actual_state.floor != visit.exit_request.to_floor
            or actual_state.portal_context is not None
            or actual_state.movement_count != visit.movement_count
            or any(x.instance_id == visit.consumed_instance_id for x in actual_state.item_instances)):
        raise ValueError('exit observation must change region without another move or restoring the used M07')
    return replace(context, active=None,
        observation_ids=context.observation_ids | {observation_id},
        completed_visit_ids=context.completed_visit_ids | {visit.visit_id},
        audit=context.audit + (SurpriseAudit(observation_id, source,
            'region_exit_confirmed', (request_id, str(actual_state.floor))),))
