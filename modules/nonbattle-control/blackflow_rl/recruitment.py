"""Actual reserve recruitment from concrete, observed recruitment tickets.

This module does not turn an aggregate ticket count into a profession, invent
temporary offers, or recruit when a ticket is merely acquired. The pinned
client's ``extraCharIds`` specifies the reserve choices guaranteed on each
ticket. Reserve operators can be recruited repeatedly; each actual ticket
creates a distinct formal operator instance with its canonical character ID
preserved in the key. Ordinary operators do not gain this exception.

Separately, the user confirmed free reserve choices for duel, resident and
ordinary-chase tickets whose profession/character identity was not observed.
Those are finite, sourced opportunity nonces, never invented item IDs or a
profession inferred from historical aggregate ticket counts.

The broker's <=5-star branch separately uses a client-confirmed common
economic action: every possible profession pair guarantees a free reserve.
Its nonce does not reveal a pair or model other character choices. This is
not evidence from the user's duel/resident/chase observations.

The caller controls when a ticket is usable and whether the player recruits.
In particular, present collectible/part rewards before offering recruitment
when implementing the user's corn-first reward order. ``ticket_instance_ids``
can restrict a reward screen to its real tickets. Storage and drop sampling
are deliberately outside this helper.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path

from .domain import EventOption, GameState, NodeType
from .operator_economy import recruit_formal


_EVIDENCE = Path(__file__).resolve().parents[1] / "data" / "evidence"


@dataclass(frozen=True, slots=True)
class BattleRecruitTicketDraw:
    item_id: str
    pool_id: str
    sample_count: int
    evidence: str = "EMPIRICAL_CONDITIONAL_TICKET_COMPOSITION"


@dataclass(frozen=True, slots=True)
class UnknownReserveOpportunity:
    """One evidenced free-reserve option, without an invented ticket/character."""
    opportunity_id: str
    source: str


BROKER_RESERVE_SOURCE = 'broker_guaranteed_reserve'
BROKER_RESERVE_EVIDENCE = 'CLIENT_CONFIRMED_BROKER_ALL_FOUR_VARIANTS_FREE_RESERVE'
BROKER_RESERVE_TICKET_IDS = tuple(f'rogue_6_recruit_ticket_5star_double_{n}_vip' for n in range(1, 5))
_UNKNOWN_RESERVE_SOURCE_LIMITS = {'duel': 2, 'resident': 1, 'normal_chase': 1,
                                  BROKER_RESERVE_SOURCE: 1}
_UNKNOWN_RESERVE_EVIDENCE = 'USER_OBSERVED_RULE:free_reserve_option'


def _unknown_reserve_evidence(source: str) -> str:
    return BROKER_RESERVE_EVIDENCE if source == BROKER_RESERVE_SOURCE else _UNKNOWN_RESERVE_EVIDENCE


def add_unknown_reserve_opportunities(engine, state: GameState, source: str, count: int) -> GameState:
    """Attach once-consumable choices to an already-settled real ticket reward.

    This does not increment the legacy ticket acquisition count. The caller
    retains its original reward quantity and source-occurrence settlement
    guard. There is no conversion from historical aggregate ticket counts.
    """
    if source not in _UNKNOWN_RESERVE_SOURCE_LIMITS:
        raise ValueError('no evidenced free reserve option for this source')
    if type(count) is not int or not 1 <= count <= _UNKNOWN_RESERVE_SOURCE_LIMITS[source]:
        raise ValueError('opportunity count exceeds the established source reward')
    if source == BROKER_RESERVE_SOURCE and any(not eligible_reserve_ids(ticket_id)
                                               for ticket_id in BROKER_RESERVE_TICKET_IDS):
        raise ValueError('broker economic abstraction requires a free reserve in every client variant')
    known_ids = {x.opportunity_id for x in state.unknown_recruit_opportunities}
    known_ids |= {key.removeprefix('observed_reserve:') for key in state.formal_operator_ids
                  if key.startswith('observed_reserve:')}
    added = tuple(UnknownReserveOpportunity(f'u{state.unknown_recruit_serial+i}', source)
                  for i in range(count))
    if any(x.opportunity_id in known_ids for x in added):
        raise ValueError('unknown recruitment nonce was already issued')
    state = replace(state,
        unknown_recruit_opportunities=state.unknown_recruit_opportunities + added,
        unknown_recruit_serial=state.unknown_recruit_serial + count,
        pending_recruit_ticket_ids=state.pending_recruit_ticket_ids + tuple(x.opportunity_id for x in added))
    for opportunity in added:
        state = engine.entry(state, 'unknown_recruit_granted', quantity=1,
            instance_id=opportunity.opportunity_id,
            source=f'{_unknown_reserve_evidence(source)}:{source}:{opportunity.opportunity_id}')
    return state


def unknown_reserve_recruit_options(
    state: GameState, *, ticket_instance_ids: frozenset[str] | None = None,
) -> tuple[EventOption, ...]:
    """Offer an actual reserve recruitment; identity/profession stay unknown."""
    return tuple(EventOption(
        f'recruit_unknown_reserve:{opportunity.opportunity_id}',
        '招募预备干员（0希望）', operation='recruit_reserve',
        instance_id=opportunity.opportunity_id, item_id=None, ends_node=False,
        description=('客户端四种职业组均保证免费预备；这里只结算共同经济动作，不虚构职业或干员身份。'
                     if opportunity.source == BROKER_RESERVE_SOURCE else
                     '实际招募机会已确认；该券的职业和预备干员身份未观测。'))
        for opportunity in state.unknown_recruit_opportunities
        if opportunity.source in _UNKNOWN_RESERVE_SOURCE_LIMITS
        and (ticket_instance_ids is None or opportunity.opportunity_id in ticket_instance_ids))


def _remove_unknown_reserve_opportunity(engine, state, opportunity_id, operation, source):
    opportunity = next((x for x in state.unknown_recruit_opportunities
                        if x.opportunity_id == opportunity_id), None)
    if opportunity is None:
        raise ValueError('no such held observed recruitment opportunity')
    state = replace(state,
        unknown_recruit_opportunities=tuple(x for x in state.unknown_recruit_opportunities
                                            if x.opportunity_id != opportunity_id),
        pending_recruit_ticket_ids=tuple(x for x in state.pending_recruit_ticket_ids if x != opportunity_id),
        stored_recruit_ticket_ids=tuple(x for x in state.stored_recruit_ticket_ids if x != opportunity_id))
    return engine.entry(state, operation, quantity=1, instance_id=opportunity_id,
        source=f'{_unknown_reserve_evidence(opportunity.source)}:{opportunity.source}:{opportunity_id}:{source}')


def discard_unknown_reserve_opportunity(engine, state: GameState, opportunity_id: str,
                                       *, source: str = 'declined_recruitment') -> GameState:
    """Discard one actual opportunity; retaining or discarding never recruits."""
    return _remove_unknown_reserve_opportunity(engine, state, opportunity_id,
                                               'unknown_recruit_discarded', source)


def resolve_unknown_reserve_recruit(engine, state: GameState, option: EventOption) -> GameState:
    """Consume one nonce and recruit once, never infer a Mechanist-eligible role."""
    if option not in unknown_reserve_recruit_options(state):
        raise ValueError('actual eligible observed reserve opportunity required')
    opportunity = next(x for x in state.unknown_recruit_opportunities
                       if x.opportunity_id == option.instance_id)
    operator_instance_id = 'observed_reserve:' + option.instance_id
    if operator_instance_id in state.formal_operator_ids:
        raise ValueError('this opportunity already recruited a reserve operator')
    state = _remove_unknown_reserve_opportunity(engine, state, option.instance_id,
                                               'unknown_recruit_consumed', 'formal_recruitment')
    return recruit_formal(engine, state, operator_instance_id,
        source=_unknown_reserve_evidence(opportunity.source), hope_cost=0)


@lru_cache(maxsize=2)
def _battle_ticket_distribution(pool_id: str) -> tuple[tuple[tuple[str, int], ...], int]:
    data = json.loads((_EVIDENCE / "rogue6_economy_observations_v1.json")
                      .read_text(encoding="utf-8"))["observed_pools"][pool_id]
    distribution = data["observed_distribution"]
    members = tuple((row["itemId"], row["occurrenceCount"])
                    for row in distribution["members"] if row["occurrenceCount"] > 0)
    if not members or any(item_id not in _ticket_records() for item_id, _ in members):
        raise ValueError("battle ticket observations contain no valid recruitment pool")
    return members, distribution["sampleCount"]


def sample_battle_recruit_ticket(rng, node_type: NodeType) -> BattleRecruitTicketDraw:
    """Choose the identity of one already-established battle ticket reward.

    The observed composition is conditional on recorded tickets, not their
    presence probability or server weights. This function never grants an
    extra ticket. Special-event and chase rewards have no supported mapping.
    The caller must not relabel such a battle as an ordinary node to reuse it.
    """
    if node_type in (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE):
        pool_id = "rogue_6:node_battle_recruit"
    elif node_type == NodeType.BATTLE_BOSS:
        pool_id = "rogue_6:node_battle_boss_recruit"
    else:
        raise ValueError("no observed recruitment ticket composition for this node type")
    members, sample_count = _battle_ticket_distribution(pool_id)
    remaining = rng.randrange(sum(weight for _, weight in members))
    for item_id, weight in members:
        if remaining < weight:
            return BattleRecruitTicketDraw(item_id, pool_id, sample_count)
        remaining -= weight
    raise AssertionError("unreachable observed ticket sampling branch")


@lru_cache(maxsize=1)
def _ticket_records() -> dict:
    return json.loads((_EVIDENCE / "rogue6_operator_economy_rules_v1.json")
                      .read_text(encoding="utf-8"))["client"]["recruit_tickets"]


@lru_cache(maxsize=1)
def _reserve_records() -> dict:
    return json.loads((_EVIDENCE / "rogue6_reserve_recruitment_v1.json")
                      .read_text(encoding="utf-8"))["reserve_characters"]


def eligible_reserve_ids(ticket_item_id: str) -> tuple[str, ...]:
    """Use explicit extra choices, including cross-profession reserve choices."""
    ticket = _ticket_records().get(ticket_item_id)
    if ticket is None:
        return ()
    reserves = _reserve_records()
    return tuple(char_id for char_id in ticket["extraCharIds"]
                 if char_id in reserves and reserves[char_id]["rarity"] in ticket["rarityList"])


def reserve_recruit_options(
    state: GameState, *, ticket_instance_ids: frozenset[str] | None = None,
) -> tuple[EventOption, ...]:
    """Return real free recruitment choices; generic ticket counts yield none."""
    result = []
    for ticket in state.item_instances:
        if ticket.category != "RECRUIT_TICKET":
            continue
        if ticket_instance_ids is not None and ticket.instance_id not in ticket_instance_ids:
            continue
        for char_id in eligible_reserve_ids(ticket.item_id):
            result.append(EventOption(
                f"recruit_reserve:{ticket.instance_id}:{char_id}",
                f"招募{_reserve_records()[char_id]['name']}（0希望，消耗此招募券）",
                operation="recruit_reserve", item_id=char_id,
                instance_id=ticket.instance_id, ends_node=False,
            ))
    return tuple(result)


def resolve_reserve_recruit(engine, state: GameState, option: EventOption) -> GameState:
    """Consume exactly one eligible instance, then perform one formal recruit.

    Validate against a fresh menu before mutation, so forged choices, reused
    ticket instances and upgrade-only tickets cannot trigger corn growth.
    The scalar ``resources.tickets`` is the legacy aggregate acquisition count;
    only removing the concrete instance authorizes/limits this transaction.
    """
    if option.operation != "recruit_reserve":
        raise ValueError("not a reserve recruitment operation")
    legal = reserve_recruit_options(state)
    if option not in legal:
        raise ValueError("reserve recruitment requires a real eligible held ticket")
    operator_instance_id = f"reserve:{option.item_id}:{option.instance_id}"
    if operator_instance_id in state.formal_operator_ids:
        raise ValueError("this ticket already created its reserve operator")
    state = engine.remove(state, option.instance_id, "consume", "reserve_recruitment")
    return recruit_formal(engine, state, operator_instance_id,
                          source="reserve_recruitment", hope_cost=0)
