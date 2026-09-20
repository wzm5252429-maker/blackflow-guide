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
from blackflow_live.vision import OCRSpan, TemplateHit, DEFAULT_MAA_ROOT, VisionPipeline, _read_image


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

    def completion(self, *, extra=(), hits=None, button_confidence=.945):
        source = [tickets()[0],OCRSpan('沉沦于树海',button_confidence,(1083,401,142,32)),*extra]
        if hits is None:
            hits = [TemplateHit('BlackFlow@Roguelike@EnterAfterRecruit.png',.923,(1091,320,135,72),.9)]
        templates = SimpleNamespace(match=lambda image,name,**kwargs: hits
                                    if name=='BlackFlow@Roguelike@EnterAfterRecruit.png' else [])
        return source,templates

    def test_initial_completion_requires_exact_enter_text_and_adjacent_marker(self):
        source,templates = self.completion()
        result = detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates)
        self.assertEqual([a.label for a in result.actions],['沉沦于树海'])
        self.assertEqual(result.actions[0].bbox,source[1].bbox)
        self.assertEqual(result.actions[0].metadata['operation'],'advance')
        self.assertEqual(result.actions[0].metadata['selection_stage'],'initial_recruitment_complete')
        self.assertNotIn('starts_battle',result.actions[0].metadata)
        self.assertNotIn('formal_operator_ids',result.evidence)

    def test_completion_with_residual_tickets_or_cross_fade_waits(self):
        for residual in (OCRSpan('特种招募券',.67,(849,321,101,25)),
                         OCRSpan('招募',.66,(877,490,46,27)),
                         OCRSpan('沉沦于树海',.99,(1085,402,142,32))):
            with self.subTest(residual=residual.text):
                source,templates = self.completion(extra=(residual,))
                result = detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates)
                self.assertFalse(result.actions)
                self.assertIn('initial_recruitment_completion_requires_observation',result.diagnostics)

    def test_completion_missing_weak_or_misaligned_marker_and_weak_text_wait(self):
        name = 'BlackFlow@Roguelike@EnterAfterRecruit.png'
        variants = [([], .99),([TemplateHit(name,.89,(1091,320,135,72),.9)],.99),
                    ([TemplateHit(name,.99,(600,320,135,72),.9)],.99),
                    ([TemplateHit(name,.99,(1091,320,135,72),.9),TemplateHit(name,.99,(1092,319,135,72),.9)],.99),
                    (None,.89)]
        for hits,confidence in variants:
            with self.subTest(hits=hits,confidence=confidence):
                source,templates = self.completion(hits=hits,button_confidence=confidence)
                result = detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates)
                self.assertFalse(result.actions)

    def test_weak_complete_enter_text_needs_two_agreeing_local_readings(self):
        source,templates = self.completion(button_confidence=.86)
        for readings,allowed in (([('沉沦于树海',.984),('沉沦于树海',.974)],True),
                                 ([('沉沦于树海',.984),('沉沦于树海',.89)],False),
                                 ([('沉沦于树海',.984),('远论于树海',.99)],False)):
            with self.subTest(readings=readings):
                self.ocr.recognize_crop.side_effect = readings
                result = detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates)
                self.assertEqual(bool(result.actions),allowed)
                if allowed:
                    self.assertEqual(result.actions[0].bbox,source[1].bbox)
                    self.assertGreaterEqual(result.actions[0].confidence,.90)

    def test_complete_initial_page_integration_and_battle_priority(self):
        source,templates = self.completion()
        templates.labels,templates.node_specs = {},[]
        pipeline = VisionPipeline(maa_root='missing',ocr=self.ocr,templates=templates,
            corridor=SimpleNamespace(score=lambda *a:[],config={'decision':{'probability_threshold':.9}}))
        pipeline._hud_resources = lambda *a:{}
        result = pipeline.analyze(self.image,source,frame_id='synthetic-initial-complete',captured_at=0)
        self.assertEqual(result.scene,'recruitment')
        self.assertEqual([a.label for a in result.actions],['沉沦于树海'])
        self.assertEqual(result.actions[0].metadata['source_frame_id'],result.frame_id)
        result = pipeline.analyze(self.image,source+[OCRSpan('开始战斗',.99,(900,630,100,40))],
                                  frame_id='synthetic-battle',captured_at=0)
        self.assertEqual(result.scene,'battle_start')
        self.assertFalse(result.actions)

    def support_detail(self, *, hits=None, footer_confidence=.801):
        source = [OCRSpan('招募助战',footer_confidence,(806,493,83,23)),
                  OCRSpan('莉娜',.694,(827,240,143,25)),OCRSpan('6',.999,(831,457,11,14))]
        if hits is None:
            hits = [TemplateHit('Return.png',.963,(17,11,128,40),.9)]
        templates = SimpleNamespace(match=lambda image,name,**kwargs: hits if name=='Return.png' else [],
                                    labels={},node_specs=[])
        self.ocr.recognize_crop.side_effect = None
        self.ocr.recognize_crop.return_value = ('招募助战',.99)
        return source,templates

    def test_support_detail_returns_using_template_without_confirming_hire(self):
        source,templates = self.support_detail()
        result = detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates)
        self.assertEqual([a.label for a in result.actions],['返回'])
        action = result.actions[0]
        self.assertEqual(action.bbox,(17,11,128,40))
        self.assertEqual(action.metadata['operation'],'event_advance')
        self.assertEqual(action.metadata['selection_stage'],'support_detail_return')
        for field in ('ends_node','operator_id','selected_operator_id','hope_cost','resource_costs'):
            self.assertNotIn(field,action.metadata)
        self.assertEqual(self.ocr.recognize_crop.call_count,2)

    def test_support_detail_ambiguous_footer_or_return_waits(self):
        variants = ('missing_return','weak_return','wrong_location','duplicate_return','inconsistent_ocr','weak_ocr','duplicate_footer')
        for defect in variants:
            with self.subTest(defect=defect):
                hit = TemplateHit('Return.png',.963,(17,11,128,40),.9)
                hits = [] if defect=='missing_return' else [hit]
                if defect=='weak_return': hits = [replace(hit,confidence=.89)]
                if defect=='wrong_location': hits = [replace(hit,bbox=(600,100,128,40))]
                if defect=='duplicate_return': hits.append(replace(hit,bbox=(30,60,128,40)))
                source,templates = self.support_detail(hits=hits)
                if defect=='duplicate_footer': source.append(replace(source[0],bbox=(808,494,83,23)))
                if defect=='inconsistent_ocr': self.ocr.recognize_crop.side_effect = [('招募助战',.99),('招募功战',.99)]
                if defect=='weak_ocr': self.ocr.recognize_crop.return_value = ('招募助战',.89)
                result = detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates)
                self.assertIsNotNone(result)
                self.assertFalse(result.actions)
                self.assertIn('support_detail_return_requires_observation',result.diagnostics)

    def test_list_choose_support_button_cannot_be_a_detail_footer(self):
        source,templates = self.support_detail()
        for label in ('选择助战','招募助战'):
            with self.subTest(label=label):
                source[0] = replace(source[0],text=label,bbox=(1030,15,130,40))
                self.assertIsNone(detect_recruitment_screen(self.image,source,ocr=self.ocr,operators=OPERATORS,templates=templates))

    def test_support_detail_occludes_background_and_battle_still_wins(self):
        source,templates = self.support_detail()
        source += [OCRSpan('卡德霍之颅',.99,(600,10,100,22)),*tickets()]
        pipeline = VisionPipeline(maa_root='missing',ocr=self.ocr,templates=templates,
            corridor=SimpleNamespace(score=lambda *a:[],config={'decision':{'probability_threshold':.9}}))
        pipeline._map = Mock(side_effect=AssertionError('support detail occludes the map'))
        pipeline._hud_resources = lambda *a:{}
        result = pipeline.analyze(self.image,source,frame_id='synthetic-support-detail',captured_at=0)
        self.assertEqual(result.scene,'recruitment')
        self.assertEqual([a.label for a in result.actions],['返回'])
        pipeline._map.assert_not_called()
        result = pipeline.analyze(self.image,source+[OCRSpan('开始战斗',.99,(900,630,100,40))],
                                  frame_id='synthetic-support-battle',captured_at=0)
        self.assertEqual(result.scene,'battle_start')
        self.assertFalse(result.actions)

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
        from blackflow_live.vision import MaaPaddleOCR, MaaTemplates
        cls.ocr = MaaPaddleOCR()
        cls.templates = MaaTemplates()

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
                result = detect_recruitment_screen(image,spans,ocr=self.ocr,operators=OPERATORS,templates=self.templates)
                self.assertIsNotNone(result)
                self.assertEqual([a.label for a in result.actions],case['expected_labels'])
                self.assertTrue(all(a.metadata['operation']!='emergency_hire' for a in result.actions))

    def test_completion_original_pixels_work_with_pil_and_opencv_lanczos(self):
        import cv2
        from PIL import Image
        original = _read_image(FIXTURES/'initial_complete_360p.png')
        variants = {'pil':np.asarray(Image.fromarray(original).resize((1280,720),Image.Resampling.LANCZOS)),
                    'opencv':cv2.resize(original,(1280,720),interpolation=cv2.INTER_LANCZOS4)}
        for name,image in variants.items():
            with self.subTest(resize=name):
                spans = self.ocr.recognize(image)
                result = detect_recruitment_screen(image,spans,ocr=self.ocr,operators=OPERATORS,templates=self.templates)
                self.assertIsNotNone(result)
                self.assertEqual([a.label for a in result.actions],['沉沦于树海'])
                self.assertGreaterEqual(result.actions[0].confidence,.90)
                original_box = next(span.bbox for span in spans if span.text=='沉沦于树海')
                self.assertEqual(result.actions[0].bbox,original_box)


if __name__ == '__main__':
    unittest.main()
