"""Precise expedition-only operator names and conflicting identity rejection."""
from types import SimpleNamespace
import unittest

from blackflow_live.vision import VisionPipeline, _merge_operator_names


class OperatorCatalogTests(unittest.TestCase):
    def test_exact_additions_keep_the_original_dictionary_unchanged(self):
        original = {'帕拉斯': {'id':'char_485_pallas','name':'帕拉斯','tags':['近战位']}}
        snapshot = {'char_485_pallas': {'name':'帕拉斯'}, 'char_509_acast': {'name':'Pith'}}
        result = _merge_operator_names(original, snapshot)
        self.assertEqual(result['Pith']['id'], 'char_509_acast')
        self.assertEqual(result['帕拉斯']['tags'], ['近战位'])
        self.assertNotIn('Pith', original)

    def test_a_conflicting_name_is_not_resurrected_by_another_record(self):
        original = {'同名': {'id':'char_first','name':'同名'}}
        snapshot = {'char_second': {'name':'同名'}, 'char_third': {'name':'同名'}}
        self.assertNotIn('同名', _merge_operator_names(original, snapshot))
        self.assertNotIn('同名', _merge_operator_names({}, snapshot))

    def test_invalid_catalog_rows_do_not_become_operator_identities(self):
        result = _merge_operator_names({}, {'wrong':{'name':'A'}, 'char_bad':{'name':None},
                                           'char_blank':{'name':' '}, 'char_other':'invalid'})
        self.assertEqual(result, {})

    def test_pinned_exclusive_operator_is_available_to_real_pipeline(self):
        pipeline = VisionPipeline(maa_root='missing', ocr=object(),
                                  templates=SimpleNamespace(), corridor=SimpleNamespace())
        self.assertEqual(pipeline.operators['Pith']['id'], 'char_509_acast')
        self.assertEqual(pipeline.operators['Stormeye']['id'], 'char_511_asnipe')
        self.assertEqual(pipeline.operators['帕拉斯']['id'], 'char_485_pallas')


if __name__ == '__main__':
    unittest.main()
