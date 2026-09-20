"""Real unselected video pixels may ground inspection, never recruitment."""
from dataclasses import replace
import unittest

from blackflow_live.models import LiveObservation
from blackflow_live.policy import action_is_safe
from blackflow_live.recruitment_cards import observed_recruitment_state
from blackflow_live.recruitment_intent import RecruitmentIntent,recruitment_fingerprint,bind_recruitment_intent
from blackflow_live.recruitment_features import RecruitmentFeatureEncoder
from tests import test_live_recruitment_cards as fixtures


class CardInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RecruitmentCardsTests.setUpClass()
        cls.fixture=fixtures.RecruitmentCardsTests()
        cls.encoder=RecruitmentFeatureEncoder()

    def observation(self,second=7.6):
        image,spans,ocr=self.fixture.frame(second)
        result=self.fixture.detect(image,spans,ocr)
        self.assertEqual(ocr.unknown_crops,0)
        actions=tuple(replace(a,metadata={**a.metadata,'source_frame_id':'saved-frame'}) for a in result.actions)
        return LiveObservation('saved-frame',0,'recruitment',result.confidence,actions,resources=result.resources,
            metadata={'image_width':1280,'image_height':720,'recruitment_state':observed_recruitment_state(result,'saved-frame')})

    def test_real_unreadable_cards_offer_only_grounded_non_spending_inspection(self):
        for second,count in ((7.6,5),(7.8,7)):
            obs=self.observation(second)
            self.assertEqual(len(obs.actions),count)
            self.assertEqual(obs.resources,{})
            for action in obs.actions:
                self.assertEqual(action.kind,'recruitment_card_inspect')
                self.assertIsNone(action.metadata['operator_id'])
                self.assertNotIn('hope_cost',action.metadata)
                self.assertTrue(action.metadata['preview_only'])
                adapted=replace(action,metadata=self.fixture.adapter.adapt(action))
                self.assertTrue(action_is_safe(adapted,obs))
            self.assertFalse(any(o['identity_observed'] for o in obs.metadata['recruitment_state']['visible_offers']))

    def test_inspect_binds_same_unknown_offer_but_recruit_does_not(self):
        obs=self.observation()
        index=next(i for i,o in enumerate(obs.metadata['recruitment_state']['visible_offers']) if o['preview_action_id'])
        intent=RecruitmentIntent('inspect',obs.frame_id,recruitment_fingerprint(obs),'synthetic-input-contract-test',.8,index)
        bound=bind_recruitment_intent(intent,obs,metadata_for=self.fixture.adapter.adapt)
        self.assertIsNotNone(bound.action)
        self.assertEqual(bound.action.metadata['selection_stage'],'card_inspect')
        self.assertIsNone(bind_recruitment_intent(replace(intent,kind='recruit',recruitment_contract='normal'),obs,metadata_for=self.fixture.adapter.adapt).action)
        features=self.encoder.encode(obs)
        self.assertFalse(features.offer_features[:,self.encoder.offer_fields.index('identity_known')].any())
        self.assertEqual(int(features.offer_features[:,self.encoder.offer_fields.index('preview_available')].sum()),len(obs.actions))
        self.assertFalse(features.offer_features[:,self.encoder.offer_fields.index('confirmation_available')].any())

    def test_marker_footer_identity_and_non_spending_contract_cannot_be_bypassed(self):
        obs=self.observation();action=obs.actions[0]
        for changes in ({'card_marker_bbox':[0,0,20,20]},{'card_marker_confidence':.89},
                        {'confirm_footer_bbox':[0,0,40,20]},{'source_frame_id':'old'},
                        {'operator_id':'char_183_skgoat'},{'selected_operator_id':'char_183_skgoat'},
                        {'hope_cost':0},{'resource_costs':{}},{'operation':'recruit_reserve'},
                        {'item_name':'随便的物品'},{'starts_battle':True}):
            raw=replace(action,metadata={**action.metadata,**changes})
            adapted=replace(raw,metadata=self.fixture.adapter.adapt(raw))
            self.assertFalse(action_is_safe(adapted,obs),changes)
        self.assertFalse(action_is_safe(replace(action,label='确认招募'),obs))
        self.assertFalse(action_is_safe(action,replace(obs,scene='battle')))

    def test_current_highlight_state_invalidates_old_inspection_candidate(self):
        obs=self.observation()
        index=next(i for i,o in enumerate(obs.metadata['recruitment_state']['visible_offers']) if o['preview_action_id'])
        offer=obs.metadata['recruitment_state']['visible_offers'][index]
        action=next(a for a in obs.actions if a.action_id==offer['preview_action_id'])
        offer['highlighted']=True
        self.assertFalse(action_is_safe(action,obs))
        intent=RecruitmentIntent('inspect',obs.frame_id,recruitment_fingerprint(obs),'synthetic-input-contract-test',.8,index)
        self.assertIsNone(bind_recruitment_intent(intent,obs,metadata_for=self.fixture.adapter.adapt).action)
        encoded=self.encoder.encode(obs)
        self.assertEqual(encoded.offer_features[index,self.encoder.offer_fields.index('preview_available')],0)

    def test_negative_card_or_footer_origin_is_rejected_even_when_right_edge_matches(self):
        obs=self.observation();action=obs.actions[0]
        for field in ('card_bbox','abandon_footer_bbox'):
            with self.subTest(field=field):
                box=list(action.metadata[field])
                right=box[0]+box[2]
                box[0],box[2]=-10,right+10
                changed=replace(action,metadata={**action.metadata,field:box})
                self.assertFalse(action_is_safe(changed,obs))

    def test_erasing_the_card_marker_removes_that_inspection(self):
        image,spans,ocr=self.fixture.frame(7.6)
        before=self.fixture.detect(image,spans,ocr)
        target=before.actions[0]
        x,y,w,h=target.metadata['card_marker_bbox']
        image[y:y+h,x:x+w]=0
        after=self.fixture.detect(image,spans,ocr)
        self.assertFalse(any(a.metadata.get('card_bbox')==target.metadata['card_bbox'] for a in after.actions))


if __name__=='__main__':unittest.main()
