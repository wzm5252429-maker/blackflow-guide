"""Concrete dedicated-ticket promotion actions with a proven zero hope cost.

The client's dedicated UPGRADE_TICKET is different from recruiting an already
held operator with a RECRUIT_TICKET. Its item description explicitly waives
hope cost; profession/rarity and actual formal-operator availability still
apply. This module and its pinned evidence belong to the runtime fingerprint.
"""
from functools import lru_cache
import json
from pathlib import Path

from .domain import EventOption
from .operator_economy import MECHANIST_ID, promote_formal


@lru_cache(maxsize=1)
def upgrade_ticket_records():
    path=Path(__file__).resolve().parents[1]/'data'/'evidence'/'rogue6_upgrade_ticket_rules_v1.json'
    return json.loads(path.read_text(encoding='utf-8'))['client']


def mechanist_eligible_upgrade_ticket_ids():
    records=upgrade_ticket_records()
    return frozenset(item_id for item_id,ticket in records['upgrade_tickets'].items()
        if 'TANK' in ticket['professionList'] and 'TIER_6' in ticket['rarityList']
        and records['items'][item_id]['type']=='UPGRADE_TICKET'
        and '不消耗希望' in records['items'][item_id]['usage'])


def mechanist_upgrade_ticket_options(state):
    if MECHANIST_ID not in state.available_formal_operator_ids or MECHANIST_ID in state.promoted_operator_ids:
        return ()
    ids=mechanist_eligible_upgrade_ticket_ids()
    records=upgrade_ticket_records()['items']
    return tuple(EventOption('mechanist_upgrade_ticket:'+item.instance_id,
        '使用'+records[item.item_id]['name']+'进阶机械师（0希望）',
        operation='mechanist_promote',item_id=MECHANIST_ID,
        instance_id=item.instance_id,ends_node=False)
        for item in state.item_instances if item.category=='UPGRADE_TICKET' and item.item_id in ids)


def resolve_mechanist_upgrade_ticket(engine,state,option):
    if option not in mechanist_upgrade_ticket_options(state):
        raise ValueError('free promotion requires an actual eligible dedicated upgrade ticket')
    # Remove only the consumed instance; acquisition/aggregate counts are not
    # a second permission to use it. Promotion is never another recruitment.
    state=engine.remove(state,option.instance_id,'consume','mechanist_upgrade_ticket')
    return promote_formal(engine,state,MECHANIST_ID,
        source='dedicated_upgrade_ticket',hope_cost=0)
