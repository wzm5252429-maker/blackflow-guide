"""Run entry and ending gates backed by client text and pinned PRTS rules.

This module does not grant a beacon, assume account unlocks, choose an operator,
or invent a sixth floor. The caller supplies actual expedition/previous-run
state and executes the returned rewards through its item ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


BEACON_ID = "rogue_6_relic_final_3"
FIRST_ENDING = "ro6_ending_1"
SECOND_ENDING = "ro6_ending_2"
THIRD_ENDING = "ro6_ending_3"
FIFTH_FLOOR_BOSSES = frozenset({"ro6_b_4", "ro6_b_4_b", "ro6_b_5"})
SIXTH_FLOOR_BOSS = "ro6_b_6"
PRTS_RULE = "B_PRTS_EXPLICIT:main_oldid_424917"


@dataclass(frozen=True, slots=True)
class RunContinuation:
    next_floor: int | None
    ending_id: str | None = None
    evidence: str = PRTS_RULE


def after_floor_clear(floor: int, *, inventory: Iterable[str],
                      stage_id: str | None = None) -> RunContinuation:
    """Call only after the real floor exit/final battle has been completed.

    Ordinary battles on floor V/VI must not call this. A supplied stage ID is
    validated so that a normal victory cannot accidentally end/extend the run.
    If the floor-V stage is unavailable, ending family 1 versus 2 is unresolved;
    the terminal gate itself remains known. A beacon wins over both V bosses.
    """
    if not 1 <= floor <= 6:
        raise ValueError("ordinary floor must be 1..6")
    if floor < 5:
        return RunContinuation(floor + 1)
    if floor == 5:
        if stage_id is not None and stage_id not in FIFTH_FLOOR_BOSSES:
            raise ValueError("not a floor-V final battle")
        if BEACON_ID in inventory:
            return RunContinuation(6)
        ending = SECOND_ENDING if stage_id == "ro6_b_5" else (
            FIRST_ENDING if stage_id in {"ro6_b_4", "ro6_b_4_b"} else None)
        return RunContinuation(None, ending)
    if stage_id is not None and stage_id != SIXTH_FLOOR_BOSS:
        raise ValueError("not a floor-VI final battle")
    return RunContinuation(None, THIRD_ENDING)


@dataclass(frozen=True, slots=True)
class SourceExpedition:
    operator_id: str
    departure_floor: int
    return_floor: int
    choice_id: str = "choice_ro6_scout_3"
    evidence: str = "A_CLIENT_CHOICE+B_PRTS_GATE"


@dataclass(frozen=True, slots=True)
class SourceExpeditionReward:
    operator_id: str
    item_id: str = BEACON_ID
    hope: int = 2
    promote_operator: bool = False
    extra_move_items: int = 0
    evidence: str = "A_CLIENT_EXPED_ENDING_RELIC+B_PRTS_RETURN"


def plan_source_expedition(*, floor: int, operator_id: str,
                           first_ending_unlocked: bool,
                           operator_is_emergency_hire: bool = False,
                           operator_already_away: bool = False) -> SourceExpedition:
    """A source expedition needs its account unlock and a selected companion."""
    if not first_ending_unlocked:
        raise ValueError("source expedition requires completed first ending")
    if floor not in (2, 3, 4):
        raise ValueError("source expedition occurs only on floors II..IV")
    if not operator_id or operator_is_emergency_hire or operator_already_away:
        raise ValueError("an available formally recruited operator is required")
    return SourceExpedition(operator_id, floor, floor + 1)


def source_expedition_return(plan: SourceExpedition, *, entered_floor: int,
                             full_technology: bool,
                             entering_black_pond: bool = False) -> SourceExpeditionReward | None:
    """Call once and remove the pending expedition on an actual ordinary return.

    A detour to Black Pond cannot return the operator early. Full technology
    adds its one random vehicle reward, not a promoted operator or an M11.
    """
    if entering_black_pond or entered_floor != plan.return_floor:
        return None
    return SourceExpeditionReward(plan.operator_id,
                                  extra_move_items=int(full_technology))


@dataclass(frozen=True, slots=True)
class StartingReward:
    choice_id: str
    name: str
    gold: int = 0
    max_hp: int = 0
    parts_capacity: int = 0
    item_category: str | None = None
    item_count: int = 0
    client_rarity_label: str | None = None
    evidence: str = "A_CLIENT_CHOICE_DESCRIPTION"


STARTING_REWARDS = (
    StartingReward("choice_ro6_startbuff_1", "未编号物", item_category="RELIC",
                   item_count=1, client_rarity_label="普通"),
    StartingReward("choice_ro6_startbuff_2", "调查预付款", gold=8),
    StartingReward("choice_ro6_startbuff_3", "空间租赁", gold=-6, parts_capacity=2),
    StartingReward("choice_ro6_startbuff_4", "退行补偿", max_hp=-2,
                   item_category="RELIC", item_count=1),
    StartingReward("choice_ro6_startbuff_5", "林间代步", item_category="MOVE", item_count=1),
    StartingReward("choice_ro6_startbuff_6", "巢寄生", parts_capacity=-1,
                   item_category="RELIC", item_count=1, client_rarity_label="稀有"),
)
STARTING_REWARD_BY_ID = {r.choice_id: r for r in STARTING_REWARDS}


@dataclass(frozen=True, slots=True)
class StartingOffer:
    choices: tuple[StartingReward, ...]
    selections: int
    evidence: str


def starting_offer(*, previous_cleared_floors: int,
                   observed_choice_ids: Iterable[str] | None = None,
                   rng: random.Random | None = None,
                   allow_synthetic: bool = False) -> StartingOffer:
    """Baseline offer only: six known choices, three shown, choose one.

    Seed-grown creatures can add offers and selection counts; those need an
    explicit cross-run profile and must not be inferred from full technology.
    Server weights and exact item pools are unknown. Observations bypass the
    synthetic uniform *offer* prior; returned item outcomes still need evidence.
    """
    if previous_cleared_floors < 0:
        raise ValueError("previous_cleared_floors cannot be negative")
    if previous_cleared_floors < 2:
        if observed_choice_ids is not None and tuple(observed_choice_ids):
            raise ValueError("baseline starting reward is unavailable before two cleared floors")
        return StartingOffer((), 0, PRTS_RULE)
    if observed_choice_ids is not None:
        ids = tuple(observed_choice_ids)
        if len(ids) != 3 or len(set(ids)) != 3 or any(i not in STARTING_REWARD_BY_ID for i in ids):
            raise ValueError("baseline observed offer must contain three distinct known choices")
        return StartingOffer(tuple(STARTING_REWARD_BY_ID[i] for i in ids), 1,
                             "OBSERVED_STARTING_OFFER")
    if not allow_synthetic or rng is None:
        raise ValueError("starting offer requires observed choices or explicit synthetic sampling")
    return StartingOffer(tuple(rng.sample(STARTING_REWARDS, 3)), 1,
                         "synthetic_uniform_starting_offer_3_of_6")
