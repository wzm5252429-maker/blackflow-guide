"""Bind a fresh neural recruitment intent to an observed UI action.

This is an execution contract, not a trained policy. It is deliberately not
installed as a replacement for CurrentNeuralPolicy. An inspect decision never
creates a persistent recruitment commitment. Every call requires a new-frame
decision; the engine still owns frame freshness, lease and physical input.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Callable

from .models import LiveObservation, ObservedAction
from .policy import action_is_safe,card_inspection_safe


@dataclass(frozen=True)
class RecruitmentIntent:
    kind: str
    source_frame_id: str
    observation_fingerprint: str
    policy_artifact_id: str
    confidence: float
    offer_index: int | None = None
    operator_id: str | None = None
    recruitment_contract: str = 'unknown'


@dataclass(frozen=True)
class RecruitmentBinding:
    action: ObservedAction | None
    reason: str


def valid_card_box(box,observation):
    if (not isinstance(box,(tuple,list)) or len(box)!=4
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box)):
        return False
    x,y,w,h=box
    width,height=observation.metadata.get('image_width'),observation.metadata.get('image_height')
    return (x>=0 and y>=0 and w>0 and h>0 and isinstance(width,(int,float)) and not isinstance(width,bool)
            and isinstance(height,(int,float)) and not isinstance(height,bool)
            and math.isfinite(width) and math.isfinite(height) and x+w<=width and y+h<=height)


def recruitment_fingerprint(observation: LiveObservation) -> str:
    """Fingerprint all current offer and action facts, excluding frame stamps.

    Frame identity is checked separately. No rounding, prior-frame carryover,
    inferred roster or default balance can hide a changed decision input.
    """
    def semantic(value):
        if isinstance(value,dict):
            return {key:semantic(item) for key,item in value.items()
                    if key not in {'source_frame_id','frame_id','captured_at'}}
        if isinstance(value,(tuple,list)):
            return [semantic(item) for item in value]
        return value
    state=observation.metadata.get('recruitment_state')
    if not isinstance(state,dict):
        raise ValueError('recruitment_state_missing')
    payload={'scene':observation.scene,'floor':observation.floor,
        'confidence':observation.confidence,'resources':observation.resources,
        'image_size':[observation.metadata.get('image_width'),observation.metadata.get('image_height')],
        'state':semantic(state),'actions':[
            {'id':a.action_id,'label':a.label,'kind':a.kind,'bbox':a.bbox,
             'enabled':a.enabled,'confidence':a.confidence,'metadata':semantic(a.metadata)}
            for a in observation.actions]}
    return sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,
        allow_nan=False,separators=(',',':')).encode('utf8')).hexdigest()


def bind_recruitment_intent(intent: RecruitmentIntent, observation: LiveObservation, *,
        metadata_for: Callable[[ObservedAction],dict]) -> RecruitmentBinding:
    """Return only a same-frame action; no input, ranking or retained plan.

    policy_artifact_id records provenance, not proof of model quality. The
    eventual policy loader must verify a trained artifact before issuing it.
    """
    try:
        return _bind(intent,observation,metadata_for=metadata_for)
    except (TypeError,ValueError,KeyError,AttributeError,OverflowError):
        return RecruitmentBinding(None,'recruitment_input_malformed')


def _bind(intent,observation,*,metadata_for):
    def wait(reason):
        return RecruitmentBinding(None,reason)
    if (not isinstance(intent,RecruitmentIntent) or not isinstance(observation,LiveObservation)
            or not isinstance(observation.metadata,dict) or not isinstance(observation.resources,dict)
            or not isinstance(intent.kind,str) or not isinstance(intent.recruitment_contract,str)
            or not all(isinstance(a,ObservedAction) and isinstance(a.metadata,dict) for a in observation.actions)):
        return wait('recruitment_input_malformed')
    if (observation.scene!='recruitment' or isinstance(observation.confidence,bool)
            or not isinstance(observation.confidence,(int,float)) or not math.isfinite(observation.confidence)
            or observation.confidence<.85):
        return wait('recruitment_scene_unverified')
    state=observation.metadata.get('recruitment_state')
    if (not isinstance(state,dict) or type(state.get('schema_version')) is not int or state.get('schema_version')!=1
            or not observation.frame_id or state.get('source_frame_id')!=observation.frame_id
            or intent.source_frame_id!=observation.frame_id):
        return wait('recruitment_frame_mismatch')
    if (not isinstance(intent.policy_artifact_id,str) or not intent.policy_artifact_id
            or isinstance(intent.confidence,bool) or not isinstance(intent.confidence,(int,float))
            or not math.isfinite(intent.confidence) or not 0<=intent.confidence<=1):
        return wait('recruitment_intent_invalid')
    try:
        if intent.observation_fingerprint!=recruitment_fingerprint(observation):
            return wait('recruitment_observation_changed')
    except (ValueError,TypeError):
        return wait('recruitment_observation_invalid')
    if intent.kind=='wait':
        return wait('recruitment_policy_wait')
    if intent.kind not in {'inspect','recruit'}:
        return wait('recruitment_intent_unsupported')
    offers=state.get('visible_offers')
    index=intent.offer_index
    if (not isinstance(offers,list) or isinstance(index,bool) or not isinstance(index,int)
            or not 0<=index<len(offers) or not all(isinstance(o,dict) for o in offers)):
        return wait('recruitment_offer_missing')
    offer=offers[index]
    def valid_box(box):return valid_card_box(box,observation)
    if not valid_box(offer.get('card_bbox')):
        return wait('recruitment_card_bounds_invalid')
    unknown_inspect=(intent.kind=='inspect' and intent.operator_id is None
        and offer.get('operator_id') is None and offer.get('identity_observed') is False)
    if not unknown_inspect and (offer.get('identity_observed') is not True or not intent.operator_id
            or offer.get('operator_id')!=intent.operator_id):
        return wait('recruitment_identity_unverified')
    if intent.kind=='recruit' and (intent.recruitment_contract not in {'normal','reserve','temporary'}
            or offer.get('recruitment_contract')!=intent.recruitment_contract):
        return wait('recruitment_contract_unverified')
    confirming=intent.kind=='recruit' and offer.get('selected_identity_verified') is True
    action_id=offer.get('confirmation_action_id' if confirming else 'preview_action_id')
    matches=[a for a in observation.actions if action_id and a.action_id==action_id]
    references=sum(bool(action_id) and (o.get('preview_action_id')==action_id
                   or o.get('confirmation_action_id')==action_id) for o in offers)
    if len(matches)!=1 or references!=1:
        return wait('recruitment_action_not_unique')
    action=matches[0]
    raw=action.metadata
    if (raw.get('source_frame_id')!=observation.frame_id
            or raw.get('operator_id')!=intent.operator_id
            or not valid_box(raw.get('card_bbox'))
            or list(raw.get('card_bbox',()))!=offer.get('card_bbox')):
        return wait('recruitment_action_identity_mismatch')
    if confirming:
        expected={'normal':'confirm_normal_recruitment','reserve':'recruit_reserve','temporary':'recruit_temporary'}
        if (raw.get('selection_stage')!='operator_confirm' or raw.get('preview_only') is not False
                or raw.get('operation')!=expected[intent.recruitment_contract]
                or raw.get('recruitment_contract')!=intent.recruitment_contract
                or raw.get('selected_operator_id')!=intent.operator_id
                or state.get('detail_operator_id')!=intent.operator_id
                or offer.get('highlighted') is not True
                or sum(o.get('highlighted') is True for o in offers)!=1
                or state.get('confirm_button_enabled_observed') is not True):
            return wait('recruitment_confirmation_mismatch')
        cost=offer.get('displayed_hope_cost')
        balance=state.get('available_hope')
        costs=raw.get('resource_costs')
        if (isinstance(cost,bool) or not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0
                or state.get('available_balance_verified') is not True
                or isinstance(balance,bool) or not isinstance(balance,(int,float)) or not math.isfinite(balance)
                or balance<cost or observation.resources.get('hope')!=balance
                or isinstance(raw.get('hope_cost'),bool) or raw.get('hope_cost')!=cost
                or not isinstance(costs,dict) or isinstance(costs.get('hope'),bool) or costs.get('hope')!=cost):
            return wait('recruitment_budget_unverified')
    elif unknown_inspect:
        if not card_inspection_safe(action,observation):
            return wait('recruitment_preview_mismatch')
    elif (action.kind!='operator_preview' or raw.get('selection_stage')!='operator_preview'
            or raw.get('preview_only') is not True or raw.get('operation')!='event'):
        return wait('recruitment_preview_mismatch')
    try:
        adapted=replace(action,metadata=metadata_for(action))
        if not action_is_safe(adapted,observation):
            return wait('recruitment_action_not_safe')
    except (TypeError,ValueError,KeyError):
        return wait('recruitment_action_not_safe')
    return RecruitmentBinding(adapted,'recruitment_confirm_bound' if confirming else 'recruitment_preview_bound')
