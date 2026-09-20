"""Observed recruitment semantics and final-confirmation masks; no game input."""
from dataclasses import replace
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from blackflow_live.models import LiveObservation, ObservedAction
from blackflow_live.policy import (
    ROOT, CurrentNeuralPolicy, ObservedMenuMetadataAdapter, action_is_safe,
)


FRAME_ID = 'recruitment-policy-fixture'
OPERATOR = 'char_485_pallas'


def ticket(profession='辅助', *, action_id='ticket', **metadata):
    return ObservedAction(action_id, '招募', 'ticket_preview', (100, 490, 80, 40), .99, metadata={
        'operation': 'select_recruit_ticket', 'ticket_name': profession + '招募券',
        'ticket_profession': profession, 'selection_stage': 'ticket_preview',
        'preview_only': True, 'grounded': True, 'source_frame_id': FRAME_ID, **metadata,
    })


def hire(operation='emergency_hire', **metadata):
    return ObservedAction('hire', '雇佣', 'ui', (900, 550, 90, 40), .99, metadata={
        'operation': operation, 'operator_id': OPERATOR, 'operator_name': '帕拉斯',
        'selected_operator_id': OPERATOR, 'selection_stage': 'operator_confirm',
        'button_enabled_observed': True, 'grounded': True, 'source_frame_id': FRAME_ID,
        'resource_costs': {'gold': 3}, **metadata,
    })


def observe(*actions, resources=None, scene='recruitment'):
    return LiveObservation(FRAME_ID, 0, scene, .99, actions=actions,
                           resources={} if resources is None else resources,
                           metadata={'image_width': 1280, 'image_height': 720})


class RecruitmentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from blackflow_rl.catalog import load_catalog
        cls.adapter = ObservedMenuMetadataAdapter(load_catalog())

    def safe(self, action, obs):
        return action_is_safe(replace(action, metadata=self.adapter.adapt(action)), obs)

    def test_same_name_ticket_preserves_both_identities_and_shared_profession(self):
        action = ticket()
        metadata = self.adapter.adapt(action)
        self.assertNotIn('item_id', metadata)
        self.assertEqual(metadata['candidate_item_ids'], [
            'rogue_6_recruit_ticket_support', 'rogue_6_recruit_ticket_support_candle'])
        self.assertTrue(metadata['identity_ambiguous'])
        self.assertEqual(metadata['ticket_professions'], ['SUPPORT'])
        self.assertEqual(metadata['category'], 'RECRUIT_TICKET')
        self.assertFalse(metadata['ticket_mechanist_eligible'])
        self.assertTrue(self.safe(action, observe(action)))
        self.assertNotIn('candidate_item_ids', action.metadata)
        self.assertNotIn('item_id', action.metadata)

    def test_tank_eligibility_requires_shared_catalog_facts_not_supplied_boolean(self):
        self.assertTrue(self.adapter.adapt(ticket('重装'))['ticket_mechanist_eligible'])
        for profession in ('先锋', '辅助', '特种'):
            action = ticket(profession, ticket_mechanist_eligible=True,
                            candidate_item_ids=['rogue_6_recruit_ticket_tank'])
            result = self.adapter.adapt(action)
            self.assertFalse(result['ticket_mechanist_eligible'])
            self.assertNotIn('rogue_6_recruit_ticket_tank', result['candidate_item_ids'])

    def test_partial_unknown_title_profession_conflict_and_wrong_id_fail_closed(self):
        for changes in ({'ticket_name': '辅助'}, {'ticket_name': '未知招募券'},
                        {'ticket_profession': '重装'}, {'ticket_profession': 'invalid'},
                        {'item_id': 'rogue_6_recruit_ticket_tank'}):
            with self.subTest(changes=changes):
                action = ticket(**changes)
                self.assertFalse(self.safe(action, observe(action)))

    def test_ticket_preview_needs_current_card_scene_and_correct_selection_stage(self):
        for changes in ({'source_frame_id': 'old'}, {'source_frame_id': None},
                        {'selection_stage': 'operator_confirm'}, {'preview_only': False}, {'grounded': False}):
            action = ticket(**changes)
            self.assertFalse(self.safe(action, observe(action)))
        action = ticket()
        self.assertFalse(self.safe(action, observe(action, scene='map')))

    def test_generic_recruitment_never_becomes_legal_from_a_button_label_alone(self):
        for operation in ('recruit', 'recruit_reserve', 'recruit_temporary', 'emergency_hire'):
            action = hire(operation)
            action = replace(action, metadata={'operation': operation, 'grounded': True, 'source_frame_id': FRAME_ID})
            self.assertFalse(self.safe(action, observe(action, resources={'gold': 50, 'hope': 8})))

    def test_final_confirmation_requires_each_identity_selection_and_button_fact(self):
        action = hire()
        for field in ('operator_id', 'operator_name', 'selected_operator_id', 'selection_stage',
                      'button_enabled_observed', 'grounded', 'source_frame_id', 'resource_costs'):
            with self.subTest(field=field):
                metadata = dict(action.metadata)
                metadata.pop(field)
                if field in ('operator_id', 'operator_name'):
                    metadata.pop('operator_id', None)
                    metadata.pop('operator_name', None)
                incomplete = replace(action, metadata=metadata)
                self.assertFalse(self.safe(incomplete, observe(incomplete, resources={'gold': 10})))

    def test_conflicting_unknown_or_unselected_operator_is_rejected(self):
        for changes in ({'operator_id': 'char_missing'}, {'operator_name': '不存在的名字'},
                        {'selected_operator_id': 'char_4230_mcnist'},
                        {'operator_id': 'char_4230_mcnist', 'selected_operator_id': 'char_4230_mcnist'},
                        {'button_enabled_observed': False}, {'button_enabled_observed': 1},
                        {'source_frame_id': 'old'}):
            with self.subTest(changes=changes):
                action = hire(**changes)
                self.assertFalse(self.safe(action, observe(action, resources={'gold': 10})))

    def test_all_final_recruit_operations_reject_preview_disguise(self):
        for operation in ('recruit', 'recruit_reserve', 'recruit_temporary', 'emergency_hire'):
            action = hire(operation, preview_only=True)
            self.assertFalse(self.safe(action, observe(action, resources={'gold': 10})))
        action = hire('event', preview_only=True, selection_stage='operator_preview')
        self.assertFalse(self.safe(action, observe(action, resources={'gold': 10})))

    def test_explicit_zero_cost_still_requires_observed_currency_balance(self):
        for operation in ('recruit', 'recruit_reserve', 'recruit_temporary', 'emergency_hire'):
            action = hire(operation, resource_costs={'hope': 0})
            self.assertFalse(self.safe(action, observe(action)))
            self.assertTrue(self.safe(action, observe(action, resources={'hope': 0})))
            self.assertFalse(self.safe(action, observe(action, resources={'hope': -1})))

    def test_missing_malformed_unaffordable_or_conflicting_costs_are_rejected(self):
        for costs in ({}, None, [], '3', {'gold': None}, {'gold': -1}, {'gold': True},
                      {'gold': float('nan')}, {'gold': 11}, {'gold': 0, 'hope': 0}):
            with self.subTest(costs=costs):
                action = hire(resource_costs=costs)
                self.assertFalse(self.safe(action, observe(action, resources={'gold': 10})))
        action = hire(resource_costs={'hope': 0}, hope_cost=2)
        self.assertFalse(self.safe(action, observe(action, resources={'hope': 10})))

    def test_affordable_hope_cost_is_grounded_without_changing_balances(self):
        action = hire('recruit_temporary', resource_costs={}, hope_cost=2)
        obs = observe(action, resources={'hope': 3})
        self.assertTrue(self.safe(action, obs))
        metadata = self.adapter.adapt(action)
        self.assertEqual(metadata['resource_delta'], {'hope': -2})
        self.assertEqual(obs.resources, {'hope': 3})
        self.assertFalse(self.safe(action, replace(obs, resources={'hope': 1})))
        self.assertFalse(self.safe(action, replace(obs, scene='map')))

    def test_real_operator_preview_remains_legal_without_cost_or_balance(self):
        action = ObservedAction('preview', '帕拉斯', 'operator_preview', (900, 120, 80, 30), .99, metadata={
            'operation': 'event', 'preview_only': True, 'operator_name': '帕拉斯',
            'selection_stage': 'operator_preview', 'grounded': True, 'source_frame_id': FRAME_ID,
        })
        obs = observe(action)
        self.assertTrue(self.safe(action, obs))
        self.assertEqual(obs.resources, {})
        self.assertNotIn('formal_operator_ids', obs.metadata)
        self.assertNotIn('available_operator_ids', obs.metadata)
        self.assertNotIn('operator_id', action.metadata)


class FrozenRecruitmentInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selection = json.loads((ROOT/'data/policies/current_neural_controller.json').read_text(encoding='utf-8'))
        if not all((ROOT/selection[key]).is_file() for key in ('checkpoint', 'menu_checkpoint', 'profile')):
            raise unittest.SkipTest('Selected local neural artifacts are unavailable')
        cls.policy = CurrentNeuralPolicy()
        if not cls.policy.load():
            raise AssertionError(cls.policy.load_error)

    def test_ambiguous_tickets_fill_only_existing_shared_features(self):
        from blackflow_rl.features import ITEM_CATEGORIES, OPTION_OPERATIONS, RESOURCE_FIELDS
        layout = self.policy.encoder.layout
        actions = tuple(ticket(p, action_id=p) for p in ('先锋', '辅助', '特种', '重装'))
        obs = observe(*actions)
        encoded, _, _ = self.policy.encoder.encode(obs, actions)
        physical = layout.option_feature_dim - len(layout.item_identities) - 12
        category_start = len(RESOURCE_FIELDS) + 4
        operation = len(RESOURCE_FIELDS) + 4 + len(ITEM_CATEGORIES) + OPTION_OPERATIONS.index('select_recruit_ticket')
        for index in range(4):
            row = encoded.option_features[index]
            # The unchanged checkpoint has no RECRUIT_TICKET category slot.
            # It must not be reinterpreted as a relic or another item category.
            self.assertFalse(row[category_start:category_start + len(ITEM_CATEGORIES)].any())
            self.assertEqual(float(row[operation]), 1)
            self.assertFalse(row[-len(layout.item_identities):].any())
            self.assertEqual(float(row[physical + 10]), float(index == 3))
            self.assertEqual(float(row[physical + 2]), .25)
        # The frozen checkpoint has no generic profession feature. Do not imply
        # it distinguishes these three professions by inventing identity bits.
        self.assertTrue((encoded.option_features[0] == encoded.option_features[1]).all())
        self.assertTrue((encoded.option_features[1] == encoded.option_features[2]).all())
        self.assertEqual(self.policy.encoder.last_coverage['ambiguous_ticket_actions'], 4)
        self.assertEqual(obs.resources, {})

    def test_frozen_menu_selects_an_actual_ambiguous_ticket_without_model_or_core_changes(self):
        files = [ROOT/'blackflow_rl/features.py', ROOT/'blackflow_rl/network.py',
                 ROOT/self.policy.selection['checkpoint'], ROOT/self.policy.selection['menu_checkpoint']]
        before = [sha256(path.read_bytes()).hexdigest() for path in files]
        actions = tuple(ticket(p, action_id=p) for p in ('先锋', '辅助', '特种'))
        obs = observe(*actions)
        with patch('blackflow_rl.simulator.BlackflowSimulator.reset', side_effect=AssertionError('no simulation')), \
             patch('blackflow_rl.simulator.BlackflowSimulator.transition', side_effect=AssertionError('no simulation')):
            decision = self.policy.select(obs)
        self.assertTrue(decision.neural, decision.reason)
        self.assertTrue(any(decision.action is action for action in actions))
        self.assertNotIn('item_id', decision.action.metadata)
        self.assertEqual(before, [sha256(path.read_bytes()).hexdigest() for path in files])

    def test_unproven_hire_is_masked_before_menu_network_selection(self):
        action = replace(hire(), metadata={'operation': 'emergency_hire', 'grounded': True})
        leave = ObservedAction('leave', '离开', 'ui', (1050, 550, 90, 40), .99,
                               metadata={'operation': 'leave', 'source_frame_id': FRAME_ID})
        obs = observe(action, leave, resources={'gold': 100})
        decision = self.policy.select(obs)
        self.assertTrue(decision.neural, decision.reason)
        self.assertIs(decision.action, leave)
        self.assertIsNone(self.policy.select(observe(action, resources={'gold': 100})).action)


if __name__ == '__main__':
    unittest.main()
