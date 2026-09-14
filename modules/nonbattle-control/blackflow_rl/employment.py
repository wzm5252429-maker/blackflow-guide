"""Deterministic EMPLOY execution from observed menus, not a candidate sampler.

The runtime bridge requires a caller to supply a real eight-character menu
and its displayed prices, and commits
the returned context and gold debit together.  No recruitment tickets, formal
operators, hope, corn appraisal, mastery bonuses, or combat wins are created.

The formal/emergency distinction follows operator_economy: recruit_formal is
not an emergency-hire primitive, because it grants permanent identity and
recruitment effects.  Integration should feed emergency_char_ids into formal
recruitment/promotion/expedition eligibility, without calling recruit_formal.

Evidence and unresolved generation/reentry/trigger rules are documented in
docs/emergency-employment-audit-2026-09-08.md.  Refreshing preserves the current
visit's three-hire limit; reentry requires a separately observed real visit.
Neither policy is evidence of free reentry or of a server-side sampling law.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable


MAX_HIRES_PER_VISIT = 3
REFRESH_COSTS = (4, 8, 12, 16)
# Identity lookup only.  This is not proof that every member can be sampled in
# every EMPLOY menu, nor an instruction to fill unknown menus from this list.
EXCLUSIVE_CHAR_IDS = frozenset({
    "char_513_apionr", "char_508_aguard", "char_511_asnipe",
    "char_509_acast", "char_510_amedic", "char_512_aprot",
    "char_504_rguard", "char_514_rdfend", "char_507_rsnipe",
    "char_506_rmedic", "char_505_rcast",
})


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty observed identifier")
    return value


def _char_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"char_\d+_[A-Za-z0-9_]+", value) is None:
        raise ValueError("a canonical observed character ID is required")
    return value


def _gold(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("gold must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class ObservedEmploymentCandidate:
    char_id: str
    displayed_price: int

    def __post_init__(self):
        _char_id(self.char_id)
        _gold(self.displayed_price)
        if self.char_id in EXCLUSIVE_CHAR_IDS and self.displayed_price != 0:
            raise ValueError("mode-exclusive emergency hires must be free")


@dataclass(frozen=True, slots=True)
class EmploymentOffer:
    offer_id: str
    char_id: str
    displayed_price: int
    claimed: bool = False


@dataclass(frozen=True, slots=True)
class EmploymentVisit:
    node_id: str
    entry_id: str
    visit_serial: int
    offers: tuple[EmploymentOffer, ...]
    evidence_source: str
    hires_this_entry: int = 0
    menu_revision: int = 0


@dataclass(frozen=True, slots=True)
class EmploymentNodeHistory:
    node_id: str
    entry_ids: tuple[str, ...]
    refreshes_used: int = 0


@dataclass(frozen=True, slots=True)
class EmergencyOperator:
    instance_id: str
    char_id: str
    source_offer_id: str
    source_node_id: str
    source_entry_id: str
    promoted: bool = True


@dataclass(frozen=True, slots=True)
class EmergencyDeparture:
    instance_id: str
    char_id: str
    battle_id: str
    source_offer_id: str


@dataclass(frozen=True, slots=True)
class EmploymentContext:
    active_visit: EmploymentVisit | None = None
    node_history: tuple[EmploymentNodeHistory, ...] = ()
    emergency_operators: tuple[EmergencyOperator, ...] = ()
    departures: tuple[EmergencyDeparture, ...] = ()
    completed_battle_ids: frozenset[str] = frozenset()
    visit_serial: int = 0
    hire_serial: int = 0

    @property
    def emergency_char_ids(self) -> frozenset[str]:
        return frozenset(operator.char_id for operator in self.emergency_operators)


@dataclass(frozen=True, slots=True)
class EmploymentTransition:
    """Commit context and gold_spent atomically; no other resource is changed."""
    context: EmploymentContext
    gold_spent: int = 0
    hired: EmergencyOperator | None = None
    departed: tuple[EmergencyDeparture, ...] = ()


def _observed_candidates(candidates: Iterable[ObservedEmploymentCandidate]):
    rows = tuple(candidates)
    if len(rows) != 8 or not all(isinstance(row, ObservedEmploymentCandidate) for row in rows):
        raise ValueError("an observed EMPLOY menu must contain exactly eight candidates")
    if len({row.char_id for row in rows}) != 8:
        raise ValueError("duplicate character IDs in an observed menu")
    if sum(row.char_id in EXCLUSIVE_CHAR_IDS for row in rows) not in (2, 3):
        raise ValueError("an observed menu must contain two or three mode-exclusive candidates")
    return rows


def _offers(rows, visit_serial: int, revision: int) -> tuple[EmploymentOffer, ...]:
    return tuple(EmploymentOffer(f"employment:{visit_serial}:{revision}:{slot}",
                                 row.char_id, row.displayed_price)
                 for slot, row in enumerate(rows))


def _history(context: EmploymentContext, node_id: str):
    return next((row for row in context.node_history if row.node_id == node_id), None)


def _replace_history(context: EmploymentContext, history: EmploymentNodeHistory):
    return tuple(row for row in context.node_history if row.node_id != history.node_id) + (history,)


def _visit(context: EmploymentContext) -> EmploymentVisit:
    if context.active_visit is None:
        raise ValueError("no active observed EMPLOY visit")
    return context.active_visit


def begin_employment_visit(
    context: EmploymentContext, *, node_id: str, entry_id: str, floor: int,
    observed_candidates: Iterable[ObservedEmploymentCandidate], evidence_source: str,
    observed_reentry: bool = False,
) -> EmploymentContext:
    """Record a real arrival/menu, never create a route action or free revisit.

    Repeated reads use active_visit, not this function.  entry_id must identify
    an actual arrival and may not be reused anywhere in this context.  The
    integration layer must independently validate movement/AP and node access.
    """
    _identifier(node_id, "node_id")
    _identifier(entry_id, "entry_id")
    _identifier(evidence_source, "evidence_source")
    if type(floor) is not int or floor not in (2, 3, 4, 5):
        raise ValueError("EMPLOY does not occur on this floor")
    if context.active_visit is not None:
        raise ValueError("close the current visit before recording another arrival")
    if any(entry_id in row.entry_ids for row in context.node_history):
        raise ValueError("this observed arrival has already been recorded")
    previous = _history(context, node_id)
    if previous is not None and observed_reentry is not True:
        raise ValueError("reentry requires evidence of another real node arrival")
    rows = _observed_candidates(observed_candidates)
    serial = context.visit_serial + 1
    history = (EmploymentNodeHistory(node_id, (entry_id,)) if previous is None
               else replace(previous, entry_ids=previous.entry_ids + (entry_id,)))
    visit = EmploymentVisit(node_id, entry_id, serial, _offers(rows, serial, 0), evidence_source)
    return replace(context, active_visit=visit, visit_serial=serial,
                   node_history=_replace_history(context, history))


def end_employment_visit(context: EmploymentContext, *, paid_refresh_cost: int = 0) -> EmploymentContext:
    visit = _visit(context)
    if type(paid_refresh_cost) is not int or paid_refresh_cost < 0:
        raise ValueError('paid refresh cost must be a nonnegative integer')
    if paid_refresh_cost:
        if paid_refresh_cost != next_employment_refresh_cost(context):
            raise ValueError('abandoned paid refresh differs from the current cost')
        history = _history(context, visit.node_id)
        context = replace(context, node_history=_replace_history(context,
            replace(history, refreshes_used=history.refreshes_used + 1)))
    return replace(context, active_visit=None)


def employment_hire_options(
    context: EmploymentContext, *, gold: int, present_char_ids: Iterable[str],
) -> tuple[EmploymentOffer, ...]:
    """Only unclaimed, affordable observed slots; present formal IDs are supplied.

    present_char_ids must be canonical IDs of the current roster, including
    formal operators.  Existing emergency identities are excluded internally.
    A present character can remain visible in a menu, but cannot be hired.
    """
    _gold(gold)
    visit = _visit(context)
    present = frozenset(_char_id(value) for value in present_char_ids) | context.emergency_char_ids
    if visit.hires_this_entry >= MAX_HIRES_PER_VISIT:
        return ()
    return tuple(row for row in visit.offers
                 if not row.claimed and row.char_id not in present and row.displayed_price <= gold)


def hire_emergency(
    context: EmploymentContext, offer_id: str, *, gold: int,
    present_char_ids: Iterable[str],
) -> EmploymentTransition:
    """Purchase one actual slot, without invoking formal recruitment effects."""
    _identifier(offer_id, "offer_id")
    allowed = employment_hire_options(context, gold=gold, present_char_ids=present_char_ids)
    offer = next((row for row in allowed if row.offer_id == offer_id), None)
    if offer is None:
        raise ValueError("offer is stale, claimed, unavailable, or unaffordable")
    visit = _visit(context)
    serial = context.hire_serial + 1
    operator = EmergencyOperator(f"emergency:{serial}", offer.char_id, offer.offer_id,
                                 visit.node_id, visit.entry_id)
    visit = replace(visit, hires_this_entry=visit.hires_this_entry + 1,
                    offers=tuple(replace(row, claimed=True) if row.offer_id == offer_id else row
                                 for row in visit.offers))
    after = replace(context, active_visit=visit, hire_serial=serial,
                    emergency_operators=context.emergency_operators + (operator,))
    return EmploymentTransition(after, gold_spent=offer.displayed_price, hired=operator)


def next_employment_refresh_cost(context: EmploymentContext) -> int | None:
    visit = _visit(context)
    history = _history(context, visit.node_id)
    if history is None:
        raise ValueError("active visit lacks its node history")
    return (REFRESH_COSTS[history.refreshes_used]
            if history.refreshes_used < len(REFRESH_COSTS) else None)


def refresh_employment(
    context: EmploymentContext, *, gold: int,
    observed_candidates: Iterable[ObservedEmploymentCandidate], evidence_source: str,
) -> EmploymentTransition:
    """Settle one paid refresh with its actually observed resulting menu.

    Future offers must not be shown to the policy before committing refresh.
    This function supplies no RNG, free peek, or speculative next menu.
    The three-hire counter is preserved, including after all three hires.
    """
    _gold(gold)
    _identifier(evidence_source, "evidence_source")
    cost = next_employment_refresh_cost(context)
    if cost is None or gold < cost:
        raise ValueError("refresh is exhausted or unaffordable")
    rows = _observed_candidates(observed_candidates)
    visit = _visit(context)
    history = _history(context, visit.node_id)
    revision = visit.menu_revision + 1
    visit = replace(visit, offers=_offers(rows, visit.visit_serial, revision),
                    menu_revision=revision, evidence_source=evidence_source)
    history = replace(history, refreshes_used=history.refreshes_used + 1)
    return EmploymentTransition(replace(context, active_visit=visit,
        node_history=_replace_history(context, history)), gold_spent=cost)


def assert_not_emergency(
    context: EmploymentContext, char_id: str, *, operation: str,
) -> None:
    """Additional guard for existing formal APIs; does not authorize an action."""
    _char_id(char_id)
    if operation not in ("formal_recruit", "formal_promotion", "expedition"):
        raise ValueError("unsupported formal operator operation")
    if char_id in context.emergency_char_ids:
        raise ValueError("an active emergency operator cannot be recruited, promoted, or dispatched")


def complete_employment_battle(
    context: EmploymentContext, *, battle_id: str,
    participating_emergency_ids: Iterable[str],
) -> EmploymentTransition:
    """Retire only actual emergency squad participants after a completed battle.

    Canonical char IDs cannot substitute for instance IDs.  An empty squad is
    valid and still records the completion nonce, preventing later reuse of
    the same battle to expire a subsequently hired operator.
    """
    _identifier(battle_id, "battle_id")
    if battle_id in context.completed_battle_ids:
        raise ValueError("battle completion has already been settled")
    participants = tuple(_identifier(value, "emergency instance ID")
                         for value in participating_emergency_ids)
    if len(participants) != len(set(participants)):
        raise ValueError("duplicate emergency participant")
    current = {operator.instance_id: operator for operator in context.emergency_operators}
    if any(instance_id not in current for instance_id in participants):
        raise ValueError("participant is not an active emergency instance")
    departed = tuple(EmergencyDeparture(instance_id, current[instance_id].char_id,
                    battle_id, current[instance_id].source_offer_id) for instance_id in participants)
    after = replace(context,
        emergency_operators=tuple(operator for operator in context.emergency_operators
                                 if operator.instance_id not in participants),
        departures=context.departures + departed,
        completed_battle_ids=context.completed_battle_ids | {battle_id})
    return EmploymentTransition(after, departed=departed)
