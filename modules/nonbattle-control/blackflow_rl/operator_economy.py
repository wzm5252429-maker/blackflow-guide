"""Confirmed formal recruitment/promotion effects; no fabricated ticket offers.

Mechanist is TIER_6 in the client character table. Full recruitment mastery
gives recruit -4, promote -2 hope, +2 capacity on recruitment, and one M11 on
promotion (also on an actual already-promoted recruitment). A temporary combat
helper is not a formal recruit. Full mastery does not make an E1 recruit E2.
"""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

from .domain import ResourceDelta
from . import relic_effects


MECHANIST_ID = 'char_4230_mcnist'


@lru_cache(maxsize=1)
def mechanist_eligible_ticket_ids():
    path=Path(__file__).resolve().parents[1]/'data'/'evidence'/'rogue6_operator_economy_rules_v1.json'
    records=json.loads(path.read_text(encoding='utf-8'))['client']['recruit_tickets']
    return frozenset(key for key,row in records.items()
        if 'TANK' in row['professionList'] and 'TIER_6' in row['rarityList'])


def mechanist_hope_cost(*, initial=False, promotion=False, full_mastery=True, difficulty=15):
    """N15 adds one to noninitial recruitment only, not to promotion.

    Client rogue_6 difficulties grade 15 and mastery recruit/upgrade entries
    are distinct rules; a free upgrade ticket bypasses this ordinary price.
    """
    if type(difficulty) is not int or not 0 <= difficulty <= 15:
        raise ValueError('difficulty must be an integer in 0..15')
    if promotion:
        return max(0, 3-(2 if full_mastery else 0))
    return max(0, 6+int(not initial and difficulty>=15)-(4 if full_mastery else 0))


def recruit_formal(engine,state,operator_id,*,source,hope_cost,already_promoted=False):
    """Called after a real eligible recruitment decision, not just a ticket drop."""
    if operator_id in state.formal_operator_ids:
        raise ValueError('already formally recruited; use promotion callback')
    if hope_cost<0 or hope_cost>state.resources.hope:
        raise ValueError('unaffordable formal recruitment')
    state=engine.apply(state,ResourceDelta(hope=-hope_cost),source)
    state=replace(state,formal_operator_ids=state.formal_operator_ids|{operator_id})
    state=engine.entry(state,'formal_recruit',quantity=1,source=source+':'+operator_id)
    state=relic_effects.on_recruit(engine,state)
    if operator_id==MECHANIST_ID and engine.config.mechanist_full_mastery:
        state=replace(state,parts_capacity=state.parts_capacity+2)
        state=engine.entry(state,'capacity_reward',quantity=2,source='rogue_6_somaster_6:recruit_reward')
    if already_promoted:
        state=promote_formal(engine,state,operator_id,source=source+':already_promoted_recruit',hope_cost=0)
    return state


def promote_formal(engine,state,operator_id,*,source,hope_cost):
    """An actual promotion or expedition reward, once for each operator."""
    if operator_id not in state.available_formal_operator_ids:
        raise ValueError('only available formal operators can be promoted')
    if operator_id in state.promoted_operator_ids:
        return state
    if hope_cost<0 or hope_cost>state.resources.hope:
        raise ValueError('unaffordable formal promotion')
    state=engine.apply(state,ResourceDelta(hope=-hope_cost),source)
    state=replace(state,promoted_operator_ids=state.promoted_operator_ids|{operator_id})
    state=engine.entry(state,'formal_promotion',quantity=1,source=source+':'+operator_id)
    if operator_id==MECHANIST_ID and engine.config.mechanist_full_mastery:
        state=engine.acquire(state,'rogue_6_scrap_M_11','rogue_6_somaster_6:upgrade_reward')
    return state
