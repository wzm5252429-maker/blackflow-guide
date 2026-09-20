"""Recruitment screen/card evidence; saved-video fixtures are not live input."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.recruitment_vision import detect_recruitment_screen
from blackflow_live.vision import OCRSpan, DEFAULT_MAA_ROOT, VisionPipeline, _read_image


FIXTURES = Path(__file__).parent/'fixtures/live_recruitment'
OPERATORS = {'帕拉斯':{'id':'char_485_pallas','name':'帕拉斯'},
             'Pith':{'id':'char_509_acast','name':'Pith'}}


def tickets():
    return [OCRSpan('初始招募',.996,(629,18,78,25)),
            OCRSpan('先锋招募券',.844,(329,321,101,25)),
            OCRSpan('辅助招募券',.994,(588,321,102,25)),
            OCRSpan('特种招募券',.912,(849,321,101,25)),
            OCRSpan('招募',.911,(356,490,47,27)),
            OCRSpan('招募',.961,(617,490,46,27)),
            OCRSpan('招募',.998,(877,490,46,27))]


def employment():
    return [OCRSpan('血色空脉',.98,(749,6,96,27)),
            OCRSpan('帕拉斯',.996,(958,125,66,26)),
            OCRSpan('Pith',.977,(1363,424,47,28)),
            OCRSpan('雇佣',.985,(1173,569,61,31)),
            OCRSpan('离开',.999,(1503,567,61,33))]


class RecruitmentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720,1280,3),np.uint8)
        self.wide_image = np.zeros((720,1589,3),np.uint8)
        self.ocr = Mock()
        self.ocr.recognize_crop.return_value = ('先锋招募券',.999)

    def detect(self, spans, *, wide=False, operators=OPERATORS):
        return detect_recruitment_screen(self.wide_image if wide else self.image,spans,
                                          ocr=self.ocr,operators=operators)

    def test_three_distinct_cards_select_tickets_without_inventing_unique_ids(self):
        result = self.detect(tickets())
        self.assertEqual([a.metadata['ticket_profession'] for a in result.actions],['先锋','辅助','特种'])
        self.assertEqual([a.bbox for a in result.actions],[s.bbox for s in tickets()[4:]])
        for action in result.actions:
            self.assertEqual(action.metadata['operation'],'select_recruit_ticket')
            self.assertEqual(action.metadata['selection_stage'],'ticket_preview')
            self.assertTrue(action.metadata['preview_only'])
            self.assertNotIn('item_id',action.metadata)
            self.assertNotIn('operator_id',action.metadata)
        self.assertEqual(self.ocr.recognize_crop.call_count,2)

    def test_missing_or_conflicting_title_cannot_borrow_another_column(self):
        for defect in ('missing','conflicting'):
            with self.subTest(defect=defect):
                source = tickets()
                if defect == 'missing':
                    source.pop(1)
                else:
                    source.append(OCRSpan('重装招募券',.99,(331,350,100,25)))
                result = self.detect(source)
                self.assertEqual([a.metadata['ticket_profession'] for a in result.actions],['辅助','特种'])

    def test_equidistant_title_does_not_choose_between_close_columns(self):
        source = [tickets()[0],OCRSpan('辅助招募券',.99,(335,320,70,25)),
                  OCRSpan('招募',.99,(300,500,60,30)),OCRSpan('招募',.99,(380,500,60,30))]
        self.assertFalse(self.detect(source).actions)

    def test_weak_title_requires_same_complete_reading_in_both_crops(self):
        self.ocr.recognize_crop.side_effect = [('先锋招募券',.99),('近卫招募券',.99)]
        self.assertEqual([a.metadata['ticket_profession'] for a in self.detect(tickets()).actions],['辅助','特种'])

    def test_prefix_title_is_not_completed_from_catalog_or_neighbor(self):
        source = tickets()
        source[1] = replace(source[1],text='先锋招募',confidence=.99)
        self.assertEqual(len(self.detect(source).actions),2)
        self.ocr.recognize_crop.assert_not_called()

    def test_duplicated_button_does_not_create_duplicate_action_or_guess_target(self):
        source = tickets()
        source.append(source[5])
        self.assertEqual([a.metadata['ticket_profession'] for a in self.detect(source).actions],['先锋','特种'])

    def test_header_in_recording_subtitle_region_cannot_create_initial_screen(self):
        source = tickets()
        source[0] = replace(source[0],bbox=(629,650,78,25))
        self.assertIsNone(self.detect(source))

    def test_emergency_footer_outputs_exact_previews_and_leave_never_generic_hire(self):
        result = self.detect(employment(),wide=True)
        self.assertEqual([a.label for a in result.actions],['帕拉斯','Pith','离开'])
        self.assertTrue(all(a.metadata['preview_only'] for a in result.actions[:-1]))
        self.assertEqual([a.metadata['operator_id'] for a in result.actions[:-1]],['char_485_pallas','char_509_acast'])
        self.assertEqual(result.actions[-1].metadata['operation'],'leave')
        self.assertTrue(result.actions[-1].metadata['ends_node'])
        self.assertTrue(all(a.metadata['operation']!='emergency_hire' for a in result.actions))
        self.assertTrue(all('resource_costs' not in a.metadata for a in result.actions))

    def test_unknown_cards_still_block_background_and_do_not_guess_names(self):
        source = [employment()[0],OCRSpan('Pith',.83,(1363,424,47,28)),
                  OCRSpan('帕拉斯',.99,(30,5,80,20)),OCRSpan('帕拉斯',.99,(600,604,90,35)),*employment()[3:]]
        result = self.detect(source,wide=True)
        self.assertEqual([a.label for a in result.actions],['离开'])
        self.ocr.recognize_crop.assert_not_called()

    def test_weak_hire_footer_with_reliable_cards_blocks_map_and_preserves_previews(self):
        source = employment()
        source[3] = replace(source[3],confidence=.89)
        pipeline = VisionPipeline(maa_root='missing',ocr=self.ocr,
            templates=SimpleNamespace(match=lambda *a,**k:[],labels={},node_specs=[]),
            corridor=SimpleNamespace(score=lambda *a:[],config={'decision':{'probability_threshold':.9}}))
        pipeline.operators = OPERATORS
        pipeline._map = Mock(side_effect=AssertionError('obscured background map must not run'))
        pipeline._hud_resources = lambda *a:{}
        result = pipeline.analyze(self.wide_image,source,frame_id='synthetic-weak-footer',captured_at=0)
        self.assertEqual(result.scene,'recruitment')
        self.assertEqual([a.label for a in result.actions],['帕拉斯','Pith','离开'])
        self.assertLessEqual(result.confidence,.89)
        self.assertFalse(result.nodes)
        self.assertTrue(all(a.metadata['operation']!='emergency_hire' for a in result.actions))
        self.assertTrue(all(a.confidence>=.90 for a in result.actions))
        pipeline._map.assert_not_called()

    def test_weak_footer_without_independent_card_evidence_waits(self):
        source = employment()
        source[3] = replace(source[3],confidence=.89)
        for variant in ('one_card','duplicate_card','overlapping_cards'):
            with self.subTest(variant=variant):
                current = [source[0],source[1],*source[3:]]
                if variant=='duplicate_card':
                    current.append(replace(source[1],bbox=(958,280,66,26)))
                elif variant=='overlapping_cards':
                    current.append(replace(source[2],bbox=source[1].bbox))
                result = self.detect(current,wide=True)
                self.assertIsNotNone(result)
                self.assertFalse(result.actions)
                self.assertIn('emergency_hire_screen_evidence_ambiguous',result.diagnostics)
        source[-1] = replace(source[-1],bbox=(1503,200,61,33))
        self.assertIsNone(self.detect(source,wide=True))

    def test_weak_footer_requires_complete_emergency_heading_for_single_card(self):
        source = employment()
        source[3] = replace(source[3],confidence=.89)
        source.pop(2)
        source.append(OCRSpan('应急雇佣的干员',.99,(795,81,170,20)))
        self.assertEqual([a.label for a in self.detect(source,wide=True).actions],['帕拉斯','离开'])
        source[-1] = replace(source[-1],text='应急雇希的干员')
        self.assertFalse(self.detect(source,wide=True).actions)

    def test_unaligned_footer_is_not_emergency_hire(self):
        source = employment()
        source[-1] = replace(source[-1],bbox=(1503,200,61,33))
        self.assertIsNone(self.detect(source,wide=True))

    def test_ambiguous_footer_waits_and_overlapping_names_do_not_create_previews(self):
        source = employment()+[OCRSpan('离开',.99,(1450,568,50,32))]
        self.assertFalse(self.detect(source,wide=True).actions)
        source = employment()+[OCRSpan('Pith',.99,(958,125,66,26))]
        self.assertEqual([a.label for a in self.detect(source,wide=True).actions],['Pith','离开'])

    def test_enumeration_order_does_not_change_grounded_actions(self):
        for source,wide in ((tickets(),False),(employment(),True)):
            with self.subTest(wide=wide):
                forward = self.detect(source,wide=wide)
                backward = self.detect(reversed(source),wide=wide)
                self.assertEqual(asdict(forward),asdict(backward))


@unittest.skipUnless((DEFAULT_MAA_ROOT/'resource/PaddleOCR/rec/inference.onnx').is_file(),
                     'Installed MAA OCR required for original-video pixel regression')
class SavedRecruitmentPixelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from blackflow_live.vision import MaaPaddleOCR
        cls.ocr = MaaPaddleOCR()

    def test_original_ticket_and_hiring_frames(self):
        from PIL import Image
        manifest = json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['current_user_native_capture'])
        for case in manifest['cases']:
            with self.subTest(case=case['image']):
                path = FIXTURES/case['image']
                self.assertEqual(sha256(path.read_bytes()).hexdigest(),case['sha256'])
                original = _read_image(path)
                height,width = original.shape[:2]
                image = np.asarray(Image.fromarray(original).resize((round(width*720/height),720),Image.Resampling.LANCZOS))
                spans = self.ocr.recognize(image)
                result = detect_recruitment_screen(image,spans,ocr=self.ocr,operators=OPERATORS)
                self.assertIsNotNone(result)
                self.assertEqual([a.label for a in result.actions],case['expected_labels'])
                self.assertTrue(all(a.metadata['operation']!='emergency_hire' for a in result.actions))


if __name__ == '__main__':
    unittest.main()
