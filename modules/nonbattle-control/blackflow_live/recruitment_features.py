"""Explicit observed inputs for a future separately trained recruitment head.

No weights, rankings, inferred prices, or synthetic team strength live here.
Unknown is represented by a separate known flag, never equated with zero.
This schema is independent of the user's frozen route/menu feature schema.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .models import LiveObservation,ObservedAction
from .policy import RECRUIT_OPERATIONS,BLOCKED_LABELS,OTHER_ENDING_LABELS,_identity_label,card_inspection_safe
from .recruitment_intent import valid_card_box


PROFESSIONS=('PIONEER','WARRIOR','TANK','SNIPER','CASTER','MEDIC','SUPPORT','SPECIAL')
RARITIES=tuple('TIER_'+str(i) for i in range(1,7))
CONTRACTS=('unknown','normal','reserve','temporary','support','promotion')
OFFER_SCALARS=('identity_known','cost_known','displayed_cost_scaled','highlighted',
    'selected_identity_verified','preview_available','confirmation_available','profession_known','rarity_known')
GLOBAL_FIELDS=('available_hope_known','available_hope_scaled','displayed_hope_known','displayed_hope_scaled',
    'all_offers_observed','visible_offer_count_scaled','roster_observed','observed_roster_count_scaled','roster_complete')


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


@dataclass(frozen=True)
class RecruitmentFeatures:
    offer_features: np.ndarray
    offer_mask: np.ndarray
    global_features: np.ndarray
    observed_roster: np.ndarray
    source_frame_id: str
    schema_sha256: str


class RecruitmentFeatureEncoder:
    def __init__(self,catalog_path: str | Path | None=None):
        path=Path(catalog_path) if catalog_path else Path(__file__).resolve().parents[1]/'data/evidence/rogue6_observed_operator_catalog_v1.json'
        raw=path.read_bytes()
        self.operators=json.loads(raw)['ordinary_and_exclusive_characters']
        self.identities=tuple(sorted(self.operators))
        self.identity_index={identity:i for i,identity in enumerate(self.identities)}
        self.offer_fields=OFFER_SCALARS+tuple('contract:'+c for c in CONTRACTS)+tuple('profession:'+p for p in PROFESSIONS)+tuple('rarity:'+r for r in RARITIES)+tuple('identity:'+i for i in self.identities)
        self.schema={'version':3,'catalog_sha256':sha256(raw).hexdigest(),'offer_fields':self.offer_fields,
            'global_fields':GLOBAL_FIELDS,'roster_identity_order':self.identities,
            'scales':{'cost':10,'hope':30,'offers':32,'roster':12}}
        self.schema_sha256=sha256(json.dumps(self.schema,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def encode(self,observation: LiveObservation) -> RecruitmentFeatures:
        if not isinstance(observation.metadata,dict) or not isinstance(observation.resources,dict):
            raise ValueError('Malformed observation containers')
        state=observation.metadata.get('recruitment_state')
        if (observation.scene!='recruitment' or not isinstance(state,dict)
                or type(state.get('schema_version')) is not int or state.get('schema_version')!=1 or not observation.frame_id
                or state.get('source_frame_id')!=observation.frame_id):
            raise ValueError('Expected same-frame recruitment offers')
        offers=state.get('visible_offers')
        if not isinstance(offers,list) or not all(isinstance(o,dict) for o in offers):
            raise ValueError('Malformed recruitment offers')
        rows=np.zeros((len(offers),len(self.offer_fields)),np.float32)
        by_field={field:i for i,field in enumerate(self.offer_fields)}
        actions={}
        for action in observation.actions:
            if not isinstance(action,ObservedAction) or not isinstance(action.metadata,dict):
                raise ValueError('Malformed observed action')
            if not isinstance(action.action_id,str) or not action.action_id:
                continue
            actions.setdefault(action.action_id,[]).append(action)
        def bound(offer,field,stage,selected=False):
            identity=offer.get('operator_id')
            ref=offer.get(field)
            known=offer.get('identity_observed') is True and isinstance(identity,str) and identity in self.operators
            unknown=offer.get('identity_observed') is False and identity is None and field=='preview_action_id'
            if (not (known or unknown) or not isinstance(ref,str) or not ref
                    or not valid_card_box(offer.get('card_bbox'),observation)):
                return False
            matches=actions.get(ref,())
            references=sum(o.get('preview_action_id')==ref or o.get('confirmation_action_id')==ref for o in offers)
            if len(matches)!=1 or references!=1:
                return False
            action=matches[0];metadata=action.metadata
            expected_stage='card_inspect' if unknown else stage
            if (action.enabled is not True or not _number(action.confidence) or not .85<=action.confidence<=1
                    or not valid_card_box(action.bbox,observation)
                    or metadata.get('grounded') is not True or metadata.get('source_frame_id')!=observation.frame_id
                    or metadata.get('operator_id')!=identity or not valid_card_box(metadata.get('card_bbox'),observation)
                    or list(metadata['card_bbox'])!=offer.get('card_bbox') or metadata.get('selection_stage')!=expected_stage
                    or metadata.get('battle') or metadata.get('starts_battle')
                    or (metadata.get('operator_name') is not None and (not known or _identity_label(metadata['operator_name'])!=_identity_label(self.operators[identity]['name'])))
                    or any(_identity_label(word).casefold() in _identity_label(action.label).casefold() for word in BLOCKED_LABELS+OTHER_ENDING_LABELS)):
                return False
            if unknown:
                return card_inspection_safe(action,observation)
            if stage=='operator_preview':
                return action.kind=='operator_preview' and metadata.get('operation')=='event' and metadata.get('preview_only') is True
            return (selected and isinstance(metadata.get('operation'),str) and metadata.get('operation') in RECRUIT_OPERATIONS and metadata.get('preview_only') is False
                and metadata.get('selected_operator_id')==identity and metadata.get('button_enabled_observed') is True)
        for row,offer in zip(rows,offers):
            identity=offer.get('operator_id')
            known=offer.get('identity_observed') is True and isinstance(identity,str) and identity in self.operators
            cost=offer.get('displayed_hope_cost')
            cost_known=_number(cost)
            if cost_known and cost/10>np.finfo(np.float32).max:
                raise ValueError('Observed cost exceeds finite feature range')
            selected=(known and offer.get('selected_identity_verified') is True and offer.get('highlighted') is True
                and state.get('detail_operator_id')==identity and valid_card_box(offer.get('card_bbox'),observation)
                and sum(o.get('highlighted') is True for o in offers)==1)
            row[:7]=(known,cost_known,cost/10 if cost_known else 0,offer.get('highlighted') is True,
                selected,
                bound(offer,'preview_action_id','operator_preview'),bound(offer,'confirmation_action_id','operator_confirm',selected))
            contract=offer.get('recruitment_contract','unknown')
            row[by_field['contract:'+(contract if contract in CONTRACTS else 'unknown')]]=1
            if known:
                record=self.operators[identity]
                row[by_field['identity:'+identity]]=1
                for name,choices in (('profession',PROFESSIONS),('rarity',RARITIES)):
                    value=record.get(name)
                    if value in choices:
                        row[by_field[name+'_known']]=1
                        row[by_field[name+':'+value]]=1
        available=state.get('available_hope')
        available_known=state.get('available_balance_verified') is True and _number(available) and _number(observation.resources.get('hope')) and observation.resources['hope']==available
        displayed=state.get('displayed_hope')
        displayed_known=_number(displayed)
        if ((available_known and available/30>np.finfo(np.float32).max)
                or (displayed_known and displayed/30>np.finfo(np.float32).max)):
            raise ValueError('Observed balance exceeds finite feature range')
        roster=observation.metadata.get('formal_operator_ids')
        roster_observed=isinstance(roster,(list,tuple))
        roster_features=np.zeros(len(self.identities),np.float32)
        if roster_observed:
            if any(not isinstance(i,str) for i in roster) or len(set(roster))!=len(roster):
                raise ValueError('Malformed observed roster')
            for identity in roster:
                if identity in self.identity_index:
                    roster_features[self.identity_index[identity]]=1
        globals_=np.array((available_known,available/30 if available_known else 0,
            displayed_known,displayed/30 if displayed_known else 0,state.get('all_offers_observed') is True,
            len(offers)/32,roster_observed,len(roster)/12 if roster_observed else 0,
            roster_observed and observation.metadata.get('formal_roster_complete') is True
                and all(identity in self.identity_index for identity in roster)),dtype=np.float32)
        return RecruitmentFeatures(rows,np.ones(len(offers),np.bool_),globals_,roster_features,observation.frame_id,self.schema_sha256)
