"""Runtime transactions for supplied temporary and emergency recruitment UI.

No automatic character or EMPLOY-menu sampler is supplied. Observations must
refer to real held tickets or the EMPLOY node actually entered. Refreshes are
paid before their replacement candidates can become observable.
"""
from dataclasses import replace
from functools import lru_cache
import json

from .domain import EventOption, NodeType, ResourceDelta
from .temporary_recruitment import (OBSERVED_CATALOG_PATH, TEMPORARY_TICKET_IDS,
    load_pinned_temporary_candidate_catalog, observe_temporary_offer)
from .employment import (EmploymentContext, begin_employment_visit,
    employment_hire_options, next_employment_refresh_cost, refresh_employment,
    complete_employment_battle)


@lru_cache(maxsize=1)
def observed_character_records():
    return json.loads(OBSERVED_CATALOG_PATH.read_text(encoding='utf-8'))['ordinary_and_exclusive_characters']


def formal_character_ids(state):
    result = {x for x in state.formal_operator_ids if x.startswith('char_')}
    result.update(x.split(':')[1] for x in state.formal_operator_ids if x.startswith('reserve:char_'))
    return frozenset(result)


def emergency_character_ids(state):
    return state.employment_context.emergency_char_ids if state.employment_context else frozenset()


def observe_temporary_recruitment(engine, state, *, ticket_instance_id,
                                  observed_char_id, observation_source):
    """Open the actually observed ticket, preserving current reward choices."""
    node = state.floor_map.node(state.current_node_id)
    ticket = next((x for x in state.item_instances if x.instance_id == ticket_instance_id), None)
    if ticket is None or ticket.item_id not in TEMPORARY_TICKET_IDS:
        raise ValueError('the observation requires a real supported held temporary ticket')
    if engine.counter(state, 'starting_reward_pending') or state.terminal:
        raise ValueError('cannot open a recruitment observation during another starting/terminal phase')
    if state.pending_node_id not in (None, node.node_id):
        raise ValueError('the active node does not match this observation')
    stored = ticket_instance_id in state.stored_recruit_ticket_ids
    if state.chase_reward_pending or state.chase_reward_context is not None:
        # The carrier's FINAL type grants no access to historical inventory.
        # Only an actual ticket in this reward phase may open a popup; the
        # engine registers newly acquired chase tickets before this callback.
        if stored or ticket_instance_id not in state.pending_recruit_ticket_ids:
            raise ValueError('chase recruitment observation requires its current pending ticket')
    if stored and node.node_type != NodeType.FINAL:
        raise ValueError('stored temporary tickets can only be opened at FINAL')
    if not stored and ticket_instance_id not in state.pending_recruit_ticket_ids:
        acquired_here = any(e.operation=='acquire' and e.instance_id==ticket_instance_id
            and e.node_id==node.node_id and e.floor==state.floor for e in state.ledger)
        if not acquired_here:
            raise ValueError('unretained historical tickets are not portable recruitment opportunities')
    # A paused ordinary reward screen, a shop, an exit or a completed reward
    # node has a known return path. Do not prematurely finish an unrelated
    # multistage story while inserting an unsolicited recruitment popup.
    reward_operations = {'take','recruit_reserve','select_recruit_ticket','mechanist_promote',
                         'retain_recruit_ticket','decline_recruitment','recruit_temporary'}
    if (node.node_id not in state.completed and node.node_type not in
            (NodeType.BATTLE_SHOP,NodeType.SCRAP_SHOP,NodeType.FINAL)
            and any(x.operation not in reward_operations for x in node.options)):
        raise ValueError('temporary observation requires the actual settled reward phase')
    state = replace(state,
        pending_recruit_ticket_ids=tuple(dict.fromkeys(state.pending_recruit_ticket_ids+(ticket_instance_id,))),
        stored_recruit_ticket_ids=tuple(x for x in state.stored_recruit_ticket_ids if x!=ticket_instance_id))
    state, offer = observe_temporary_offer(engine,state,load_pinned_temporary_candidate_catalog(),
        ticket_instance_id=ticket_instance_id,observed_char_id=observed_char_id,
        observation_source=observation_source,blocked_operator_ids=emergency_character_ids(state))
    state = replace(state, temporary_recruit_offers=state.temporary_recruit_offers+(offer,),
                    pending_node_id=node.node_id)
    return engine.after_recruitment_choice(state,node)


def _employment_node(state):
    if state.chase_reward_pending or state.chase_reward_context is not None:
        raise ValueError('off-map chase rewards cannot open the suspended employment node')
    node = state.floor_map.node(state.current_node_id)
    if (node.node_type != NodeType.EMPLOY or state.pending_node_id != node.node_id
            or state.terminal):
        raise ValueError('an actually entered active EMPLOY node is required')
    return node


def _validate_employment_characters(candidates):
    rows = tuple(candidates)
    known = observed_character_records()
    if any(x.char_id not in known for x in rows):
        raise ValueError('EMPLOY candidate is absent from the pinned character client')
    return rows


def employment_menu(engine, state):
    node = _employment_node(state)
    context = state.employment_context
    if context is None or context.active_visit is None:
        return engine.set_options(state,node,(
            EventOption('observe_employment','查看实际应急雇佣候选',operation='needs_observation',ends_node=False),
            EventOption('leave','离开',operation='leave')))
    if engine.counter(state, node.node_id+':employment_refresh_paid'):
        return engine.set_options(state,node,(
            EventOption('observe_employment_refresh','读取已支付刷新的实际候选',operation='needs_observation',ends_node=False),
            EventOption('leave','离开',operation='leave')),observation=True)
    records = observed_character_records()
    options = [EventOption(offer.offer_id,
        f'应急雇佣{records[offer.char_id]["name"]}（{offer.displayed_price}锭）',
        operation='emergency_hire',item_id=offer.char_id,price=offer.displayed_price,
        effect=ResourceDelta(gold=-offer.displayed_price),ends_node=False)
        for offer in employment_hire_options(context,gold=state.resources.gold,
                                             present_char_ids=formal_character_ids(state))]
    cost = next_employment_refresh_cost(context)
    if cost is not None:
        options.append(EventOption('employment_refresh','刷新实际雇佣候选',
            operation='employment_refresh',price=cost,effect=ResourceDelta(gold=-cost),ends_node=False))
    options.append(EventOption('leave','离开',operation='leave'))
    return engine.set_options(state,node,options)


def observe_employment_menu(engine, state, *, observed_candidates, observation_source):
    node = _employment_node(state)
    context = state.employment_context or EmploymentContext()
    if engine.counter(state,node.node_id+':employment_refresh_paid'):
        raise ValueError('use the committed-refresh observation entry point')
    context = begin_employment_visit(context,node_id=node.node_id,
        entry_id=f'{state.floor}:{node.node_id}:move:{state.movement_count}',floor=state.floor,
        observed_candidates=_validate_employment_characters(observed_candidates),
        evidence_source=observation_source,
        observed_reentry=any(x.node_id==node.node_id for x in context.node_history))
    state = replace(state, employment_context=context)
    state = engine.entry(state,'employment_menu_observed',quantity=8,
        source='OBSERVED_MENU:'+observation_source)
    return employment_menu(engine,state)


def observe_employment_refresh(engine, state, *, observed_candidates, observation_source):
    node = _employment_node(state)
    paid = engine.counter(state,node.node_id+':employment_refresh_paid')
    if not paid or state.employment_context is None:
        raise ValueError('refresh must be committed before its candidates are observed')
    result = refresh_employment(state.employment_context,gold=state.resources.gold+paid,
        observed_candidates=_validate_employment_characters(observed_candidates),evidence_source=observation_source)
    if result.gold_spent != paid:
        raise ValueError('observed refresh cost differs from the already paid amount')
    state = replace(state,employment_context=result.context)
    state = engine.set_counter(state,node.node_id+':employment_refresh_paid',0)
    state = engine.entry(state,'employment_menu_observed',quantity=8,
        source='OBSERVED_REFRESH:'+observation_source)
    return employment_menu(engine,state)


def observe_emergency_battle_participants(engine,state,*,battle_number,participating_emergency_ids):
    """Apply a supplied roster only to an already completed battle counter.

    No default combat roster or automatic departures are assumed. An operator
    hired after this battle must not be retroactively assigned to it.
    """
    if type(battle_number) is not int or not 1 <= battle_number <= state.battle_count:
        raise ValueError('an actual completed battle number is required')
    if state.employment_context is None:
        raise ValueError('there is no observed emergency roster')
    participating_emergency_ids=tuple(participating_emergency_ids)
    for instance_id in participating_emergency_ids:
        if engine.counter(state,'emergency_hired_after_battle:'+instance_id) >= battle_number:
            raise ValueError('operator was hired after this battle')
    result = complete_employment_battle(state.employment_context,
        battle_id=f'battle:{battle_number}',participating_emergency_ids=participating_emergency_ids)
    state = replace(state,employment_context=result.context)
    for departed in result.departed:
        state = engine.entry(state,'emergency_departure',quantity=1,
            instance_id=departed.instance_id,source=f'OBSERVED_PARTICIPANT:battle:{battle_number}:{departed.char_id}')
    return state
