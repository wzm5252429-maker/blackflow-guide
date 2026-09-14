"""Commander progression from the client level table and published settlements.

The exp column is the cost to reach that level from its predecessor, not a
cumulative threshold. Full technology gives three separate 10% multipliers.
Base battle rewards: https://prts.wiki/w/沉沦者的黑流树海
Level/technology source: source_data/roguelike_topic_table_full.json, rogue_6.
Combat victory is still supplied by the economic simulator's declared oracle.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from .domain import NodeType, ResourceDelta
from . import relic_effects

# (experience required, hope reward, squad capacity reward), levels 1..10.
LEVELS = ((0,0,0),(10,4,0),(24,4,0),(36,5,1),(40,5,0),
          (55,6,1),(65,6,0),(75,6,0),(90,6,1),(95,8,0))
NORMAL_EXP = (10,12,13,15,20,20)
ELITE_EXP = (12,18,25,30,36,36)
BOSS_EXP = (0,0,32,0,50,70)


def battle_base_exp(node,floor):
    if node.stage_id:
        from .events import SPECIAL_BATTLES
        from .residents import RESIDENT_BATTLES
        if node.stage_id in SPECIAL_BATTLES:
            return SPECIAL_BATTLES[node.stage_id]['command_exp']
        if node.stage_id in RESIDENT_BATTLES:
            return RESIDENT_BATTLES[node.stage_id].command_exp
    if node.node_type==NodeType.BATTLE_BOSS:
        return BOSS_EXP[floor-1]
    if node.node_type in (NodeType.BATTLE_ELITE,NodeType.BATTLE_SAVAGE):
        return ELITE_EXP[floor-1]
    return NORMAL_EXP[floor-1]


def award_battle_exp(engine,state,base_exp,source='battle'):
    multiplier=Fraction(11,10)**3 if engine.config.full_tech else Fraction(1)
    for _,up in relic_effects.reward_up_parameters(engine,state,'rogue_6_exp'):
        multiplier *= 1+Fraction(str(up))
    amount=int(base_exp*multiplier)
    level=max(1,engine.counter(state,'commander_level'))
    exp=engine.counter(state,'commander_exp')+amount
    state=engine.entry(state,'commander_exp',quantity=amount,source=source)
    while level<10 and exp>=LEVELS[level][0]:
        required,hope,capacity=LEVELS[level]
        exp-=required
        level+=1
        state=engine.apply(state,ResourceDelta(hope=hope),'commander_level')
        state=replace(state,squad_capacity=state.squad_capacity+capacity)
        state=engine.entry(state,'commander_level',quantity=level,source=source)
    state=engine.set_counter(state,'commander_exp',exp)
    return relic_effects.on_commander_level(engine,state,level)
