"""Inputs only, no new weights or claim of a trained recruitment policy."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np

from blackflow_live.recruitment_features import RecruitmentFeatureEncoder,GLOBAL_FIELDS
from tests import test_live_recruitment_intent as intent_support


class RecruitmentFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        intent_support.RecruitmentIntentTests.setUpClass()
        cls.fixtures=intent_support.RecruitmentIntentTests()
        cls.encoder=RecruitmentFeatureEncoder()

    def obs(self):return self.fixtures.observation()

    def test_observed_identity_rows_are_distinct_without_changing_frozen_layout(self):
        obs=self.obs()
        offers=obs.metadata['recruitment_state']['visible_offers']
        a=next(i for i,o in enumerate(offers) if o['operator_id']=='char_183_skgoat')
        b=next(i for i,o in enumerate(offers) if o['operator_id']=='char_278_orchid')
        encoded=self.encoder.encode(obs)
        offset=self.encoder.offer_fields.index('identity:char_183_skgoat')
        self.assertEqual(encoded.offer_features[a,offset],1)
        self.assertEqual(encoded.offer_features[b,offset],0)
        self.assertFalse(np.array_equal(encoded.offer_features[a],encoded.offer_features[b]))
        self.assertEqual(encoded.schema_sha256,self.encoder.schema_sha256)

    def test_zero_unknown_and_padding_are_separate(self):
        obs=self.obs()
        offers=obs.metadata['recruitment_state']['visible_offers']
        index=next(i for i,o in enumerate(offers) if o['displayed_hope_cost']==0)
        known=self.encoder.encode(obs)
        changed=deepcopy(obs)
        changed.metadata['recruitment_state']['visible_offers'][index]['displayed_hope_cost']=None
        unknown=self.encoder.encode(changed)
        cost_known=self.encoder.offer_fields.index('cost_known')
        self.assertEqual(known.offer_features[index,cost_known],1)
        self.assertEqual(unknown.offer_features[index,cost_known],0)
        self.assertEqual(len(unknown.offer_mask),len(offers))
        self.assertTrue(unknown.offer_mask.all())
        self.assertTrue(any(o['operator_id'] is None for o in offers))

    def test_reordering_offers_reorders_rows_without_losing_context(self):
        obs=self.obs()
        before=self.encoder.encode(obs)
        changed=deepcopy(obs)
        changed.metadata['recruitment_state']['visible_offers'].reverse()
        after=self.encoder.encode(changed)
        np.testing.assert_array_equal(after.offer_features,before.offer_features[::-1])
        np.testing.assert_array_equal(after.global_features,before.global_features)

    def test_catalog_free_flag_does_not_supply_an_unreadable_cost(self):
        obs=self.obs()
        offer=next(o for o in obs.metadata['recruitment_state']['visible_offers'] if o['operator_id']=='char_183_skgoat')
        offer['displayed_hope_cost']=None
        row=self.encoder.encode(obs).offer_features[obs.metadata['recruitment_state']['visible_offers'].index(offer)]
        self.assertEqual(row[self.encoder.offer_fields.index('cost_known')],0)
        self.assertEqual(row[self.encoder.offer_fields.index('displayed_cost_scaled')],0)

    def test_displayed_balance_does_not_become_spendable_or_a_known_roster(self):
        obs=self.obs()
        obs.metadata['recruitment_state']['available_balance_verified']=False
        encoded=self.encoder.encode(obs)
        self.assertEqual(encoded.global_features[GLOBAL_FIELDS.index('available_hope_known')],0)
        self.assertEqual(encoded.global_features[GLOBAL_FIELDS.index('displayed_hope_known')],1)
        self.assertEqual(encoded.global_features[GLOBAL_FIELDS.index('roster_observed')],0)
        self.assertFalse(encoded.observed_roster.any())
        obs.metadata['formal_operator_ids']=[]
        encoded=self.encoder.encode(obs)
        self.assertEqual(encoded.global_features[GLOBAL_FIELDS.index('roster_observed')],1)
        self.assertEqual(encoded.global_features[GLOBAL_FIELDS.index('roster_complete')],0)

    def test_wrong_frame_and_wrong_scene_are_not_training_inputs(self):
        obs=self.obs()
        for changed in (replace(obs,frame_id='other'),replace(obs,scene='battle')):
            with self.assertRaises(ValueError):self.encoder.encode(changed)

    def test_unreliable_confirmation_references_are_not_available(self):
        for change in ({'operation':'start_battle'},{'preview_only':True},{'operator_name':'梓兰'},
                       {'card_bbox':None},{'card_bbox':[]}):
            obs=self.obs()
            target=next(a for a in obs.actions if a.label=='确认招募')
            target.metadata.update(change)
            rows=self.encoder.encode(obs).offer_features
            self.assertFalse(rows[:,self.encoder.offer_fields.index('confirmation_available')].any())
        obs=self.obs()
        offers=obs.metadata['recruitment_state']['visible_offers']
        offers.append(deepcopy(next(o for o in offers if o['confirmation_action_id'])))
        rows=self.encoder.encode(obs).offer_features
        self.assertFalse(rows[:,self.encoder.offer_fields.index('confirmation_available')].any())

    def test_unverified_selection_is_not_a_confirmation(self):
        obs=self.obs()
        next(o for o in obs.metadata['recruitment_state']['visible_offers'] if o['confirmation_action_id'])['selected_identity_verified']=False
        rows=self.encoder.encode(obs).offer_features
        self.assertFalse(rows[:,self.encoder.offer_fields.index('confirmation_available')].any())

    def test_unknown_roster_identity_prevents_claiming_complete_roster(self):
        obs=self.obs()
        obs.metadata.update(formal_operator_ids=['unknown-character'],formal_roster_complete=True)
        result=self.encoder.encode(obs)
        self.assertEqual(result.global_features[GLOBAL_FIELDS.index('roster_complete')],0)
        self.assertEqual(result.global_features[GLOBAL_FIELDS.index('roster_observed')],1)
        self.assertGreater(result.global_features[GLOBAL_FIELDS.index('observed_roster_count_scaled')],0)


if __name__=='__main__':unittest.main()
