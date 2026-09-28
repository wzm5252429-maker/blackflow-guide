"""Intent binding only; synthetic normal contracts never claim pixel proof."""
from copy import deepcopy
from dataclasses import asdict,replace
import unittest

from blackflow_live.models import LiveObservation
from blackflow_live.recruitment_cards import observed_recruitment_state
from blackflow_live.recruitment_intent import RecruitmentIntent,bind_recruitment_intent,recruitment_fingerprint
from tests import test_live_recruitment_cards as fixture_support


class RecruitmentIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_support.RecruitmentCardsTests.setUpClass()
        cls.fixture=fixture_support.RecruitmentCardsTests()

    def observation(self, *, normal_contract=False):
        screen=self.fixture.detect(*self.fixture.frame(8))
        actions=tuple(replace(a,metadata={**a.metadata,'source_frame_id':'frame'}) for a in screen.actions)
        state=observed_recruitment_state(screen,'frame')
        if normal_contract:
            # Explicit synthetic contract enrichment tests the future interface.
            # The original frame/detector does not establish normal vs reserve.
            for offer in state['visible_offers']:
                offer['recruitment_contract']='normal'
            actions=tuple(replace(a,metadata={**a.metadata,'recruitment_contract':'normal',
                'operation':'confirm_normal_recruitment' if a.metadata['selection_stage']=='operator_confirm' else a.metadata['operation']}) for a in actions)
        return LiveObservation('frame',123,'recruitment',screen.confidence,actions,
            resources=screen.resources,metadata={'image_width':1280,'image_height':720,'recruitment_state':state})

    def intent(self,obs,kind='recruit',identity='char_183_skgoat',contract='normal'):
        index=next(i for i,o in enumerate(obs.metadata['recruitment_state']['visible_offers']) if o['operator_id']==identity)
        return RecruitmentIntent(kind,obs.frame_id,recruitment_fingerprint(obs),'synthetic-test-artifact',.8,index,identity,contract)

    def bind(self,intent,obs):
        before=deepcopy(asdict(obs))
        result=bind_recruitment_intent(intent,obs,metadata_for=self.fixture.adapter.adapt)
        self.assertEqual(before,asdict(obs),'binding mutated the observed game state')
        return result

    def test_real_preview_maps_to_inspect_without_acquiring_an_operator(self):
        obs=self.observation()
        intent=self.intent(obs,'inspect','char_278_orchid','unknown')
        result=self.bind(intent,obs)
        self.assertEqual(result.reason,'recruitment_preview_bound')
        self.assertEqual(result.action.action_id,next(a.action_id for a in obs.actions if a.label=='梓兰'))
        self.assertTrue(result.action.metadata['preview_only'])

    def test_explicit_normal_contract_uses_only_current_same_identity_confirmation(self):
        obs=self.observation(normal_contract=True)
        result=self.bind(self.intent(obs),obs)
        self.assertEqual(result.reason,'recruitment_confirm_bound')
        self.assertEqual(result.action.action_id,next(a.action_id for a in obs.actions if a.label=='确认招募'))
        self.assertEqual(result.action.metadata['operation'],'confirm_normal_recruitment')

    def test_inspect_never_becomes_confirmation_when_target_is_already_selected(self):
        obs=self.observation(normal_contract=True)
        self.assertIsNone(self.bind(self.intent(obs,'inspect'),obs).action)

    def test_unknown_contract_or_old_reserve_operation_cannot_impersonate_normal(self):
        obs=self.observation()
        self.assertEqual(self.bind(self.intent(obs),obs).reason,'recruitment_contract_unverified')
        obs=deepcopy(obs)
        for offer in obs.metadata['recruitment_state']['visible_offers']:
            offer['recruitment_contract']='normal'
        self.assertIsNone(self.bind(self.intent(obs),obs).action)

    def test_frame_and_fingerprint_changes_invalidate_intent(self):
        original=self.observation(normal_contract=True)
        intent=self.intent(original)
        self.assertIsNone(self.bind(replace(intent,source_frame_id='old-frame'),original).action)
        for field,value in (('displayed_hope_cost',1),('operator_id','char_278_orchid'),
                            ('selected_identity_verified',False),('card_bbox',[0,0,100,100])):
            obs=deepcopy(original)
            obs.metadata['recruitment_state']['visible_offers'][intent.offer_index][field]=value
            self.assertEqual(self.bind(intent,obs).reason,'recruitment_observation_changed')
        changed=replace(original,resources={'hope':3})
        self.assertEqual(self.bind(intent,changed).reason,'recruitment_observation_changed')

    def test_missing_cross_card_duplicate_or_wrong_frame_action_cannot_bind(self):
        original=self.observation(normal_contract=True)
        target=next(a for a in original.actions if a.label=='确认招募')
        for kind in ('missing','other-card','old-frame','duplicate','different-identity'):
            obs=deepcopy(original)
            if kind=='missing':obs=replace(obs,actions=tuple(a for a in obs.actions if a!=target))
            elif kind=='duplicate':obs=replace(obs,actions=obs.actions+(target,))
            else:
                metadata=dict(target.metadata)
                if kind=='other-card':metadata['card_bbox']=[0,0,100,100]
                elif kind=='old-frame':metadata['source_frame_id']='old-frame'
                else:metadata['operator_id']='char_278_orchid'
                obs=replace(obs,actions=tuple(replace(a,metadata=metadata) if a==target else a for a in obs.actions))
            with self.subTest(kind=kind):
                self.assertIsNone(self.bind(self.intent(obs),obs).action)

    def test_highlight_and_displayed_balance_do_not_prove_affordable_confirmation(self):
        original=self.observation(normal_contract=True)
        for field,value in (('available_balance_verified',False),('available_hope',None),
                            ('available_hope',-1),('confirm_button_enabled_observed',False),
                            ('detail_operator_id','char_278_orchid')):
            obs=deepcopy(original)
            obs.metadata['recruitment_state'][field]=value
            self.assertIsNone(self.bind(self.intent(obs),obs).action)
        for value in (None,True,float('inf'),1):
            obs=deepcopy(original)
            intent=self.intent(obs)
            obs.metadata['recruitment_state']['visible_offers'][intent.offer_index]['displayed_hope_cost']=value
            if value==float('inf'):
                self.assertIsNone(self.bind(intent,obs).action)
            else:self.assertIsNone(self.bind(self.intent(obs),obs).action)

    def test_battle_ending_unknown_or_incomplete_intents_produce_no_action(self):
        obs=self.observation(normal_contract=True)
        intent=self.intent(obs)
        for scene in ('battle','battle_start','ending','ending_complete','unknown','loading','map'):
            self.assertIsNone(self.bind(intent,replace(obs,scene=scene)).action)
        for change in ({'offer_index':True},{'offer_index':-1},{'confidence':float('nan')},
                       {'policy_artifact_id':''},{'kind':'decline'}):
            self.assertIsNone(self.bind(replace(intent,**change),obs).action)
        self.assertEqual(self.bind(replace(intent,kind='wait'),obs).reason,'recruitment_policy_wait')

    def test_recruit_may_preview_but_does_not_persist_a_confirmation_plan(self):
        obs=self.observation(normal_contract=True)
        intent=self.intent(obs,'recruit','char_278_orchid')
        result=self.bind(intent,obs)
        self.assertEqual(result.reason,'recruitment_preview_bound')
        next_obs=replace(obs,frame_id='next-frame')
        self.assertIsNone(self.bind(intent,next_obs).action)

    def test_two_offers_cannot_share_one_confirmation_reference(self):
        obs=self.observation(normal_contract=True)
        intent=self.intent(obs)
        offers=obs.metadata['recruitment_state']['visible_offers']
        duplicate=deepcopy(offers[intent.offer_index])
        duplicate['highlighted']=False
        offers.append(duplicate)
        self.assertEqual(self.bind(self.intent(obs),obs).reason,'recruitment_action_not_unique')

    def test_boolean_cost_cannot_be_hidden_by_equal_numeric_hope_cost(self):
        from blackflow_live.policy import action_is_safe
        obs=self.observation(normal_contract=True)
        action=next(a for a in obs.actions if a.label=='确认招募')
        action.metadata['resource_costs']={'hope':False}
        adapted=replace(action,metadata=self.fixture.adapter.adapt(action))
        self.assertFalse(action_is_safe(adapted,obs))
        self.assertIsNone(self.bind(self.intent(obs),obs).action)

    def test_matching_invalid_card_rectangles_are_not_binding_evidence(self):
        for box in ([],[1,2],[-1,0,100,20],[1,1,0,20],[1,1,True,20],[0,0,1500,200]):
            obs=self.observation(normal_contract=True)
            intent=self.intent(obs)
            obs.metadata['recruitment_state']['visible_offers'][intent.offer_index]['card_bbox']=box
            next(a for a in obs.actions if a.label=='确认招募').metadata['card_bbox']=box
            self.assertIsNone(self.bind(self.intent(obs),obs).action)

    def test_malformed_intent_or_observation_returns_wait(self):
        obs=self.observation(normal_contract=True)
        intent=self.intent(obs)
        for change in ({'kind':[]},{'recruitment_contract':[]}):
            self.assertIsNone(self.bind(replace(intent,**change),obs).action)
        for change in ({'metadata':None},{'resources':None}):
            self.assertIsNone(self.bind(intent,replace(obs,**change)).action)
        for version in (True,1.):
            changed=deepcopy(obs)
            changed.metadata['recruitment_state']['schema_version']=version
            self.assertIsNone(self.bind(self.intent(changed),changed).action)


if __name__=='__main__':unittest.main()
