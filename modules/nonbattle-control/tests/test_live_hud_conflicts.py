"""Independent synthetic HUD conflict checks; no current screen capture or OCR."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.vision import OCRSpan
from tests.test_live_vision import observer


class IndependentHudConflictTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720, 1246, 3), np.uint8)
        self.floor = OCRSpan('玻利瓦尔肤层', .99, (500, 4, 110, 24))
        self.life = OCRSpan('目标生命值', .99, (153, 11, 70, 15))

    def test_analyze_cannot_restore_disputed_inline_or_hud_values(self):
        cases = {
            'hp_conflict_plus_inline': [
                OCRSpan('目标生命值4/4', .99, (10, 10, 110, 20)), self.life,
                OCRSpan('3/4', .99, (155, 31, 31, 19)),
                OCRSpan('4/4', .99, (155, 31, 31, 19))],
            'impossible_inline_plus_valid_hud': [
                OCRSpan('目标生命值5/4', .99, (10, 10, 110, 20)), self.life,
                OCRSpan('4/4', .99, (155, 31, 31, 19))],
            'zero_denominator_plus_valid_hud': [
                OCRSpan('目标生命值0/0', .99, (10, 10, 110, 20)), self.life,
                OCRSpan('4/4', .99, (155, 31, 31, 19))],
            'different_valid_inline_and_hud': [
                OCRSpan('目标生命值3/4', .99, (10, 10, 110, 20)), self.life,
                OCRSpan('4/4', .99, (155, 31, 31, 19))],
            'relic_conflict_plus_inline': [
                OCRSpan('收藏品1', .99, (20, 640, 80, 24)),
                OCRSpan('收藏品', .99, (128, 688, 58, 23)),
                OCRSpan('1', .99, (148, 660, 13, 20)),
                OCRSpan('7', .99, (148, 660, 13, 20))],
            'inline_conflicts': [
                OCRSpan('目标生命值4/4', .99, (10, 10, 110, 20)),
                OCRSpan('生命值4/6', .99, (300, 10, 110, 20))],
        }
        for name, spans in cases.items():
            for reverse in (False, True):
                with self.subTest(case=name, reverse=reverse):
                    ordered = list(reversed(spans)) if reverse else spans
                    result = observer().analyze(
                        self.image, [self.floor, *ordered],
                        frame_id='independent-synthetic', captured_at=1.0)
                    fields = ('relics',) if name.startswith('relic') else ('hp', 'max_hp')
                    for field in fields:
                        self.assertNotIn(field, result.resources)
                        self.assertIn('resource_reading_conflict:' + field, result.diagnostics)

    def test_high_confidence_crop_conflicts_remain_unknown(self):
        weak = OCRSpan('4/4', .8, (155, 31, 31, 19))
        # Crop OCR is mocked, but the real numeric_crop preprocessing,
        # aggregation, HUD fallback and analyze merge paths execute.
        cases = {
            'two_tight_crops_disagree': (
                [self.life, weak, weak],
                [('3/4', .99)] * 3 + [('4/4', .99)] * 6),
            'tight_preprocessing_disagrees_then_wide_recrop': (
                [self.life, weak],
                [('3/4', .99), ('4/4', .99), ('4/4', .99)] + [('4/4', .99)] * 3),
            'impossible_tight_crop_then_second_label': (
                [self.life, weak,
                 OCRSpan('生命值', .99, (400, 11, 70, 15)),
                 OCRSpan('4/4', .99, (402, 31, 31, 19))],
                [('5/4', .99)] * 3),
            'relic_preprocessing_disagrees_then_second_label': (
                [OCRSpan('收藏品', .99, (128, 688, 58, 23)),
                 OCRSpan('藏品', .99, (400, 688, 58, 23)),
                 OCRSpan('1', .99, (420, 660, 13, 20))],
                [('1', .99), ('7', .99)]),
        }
        for name, (spans, readings) in cases.items():
            with self.subTest(case=name):
                pipeline = observer()
                pipeline.ocr = Mock()
                pipeline.ocr.recognize_crop.side_effect = readings
                result = pipeline.analyze(
                    self.image, [self.floor, *spans],
                    frame_id='independent-synthetic', captured_at=1.0)
                fields = ('relics',) if name.startswith('relic') else ('hp', 'max_hp')
                for field in fields:
                    self.assertNotIn(field, result.resources)
                    self.assertIn('resource_reading_conflict:' + field, result.diagnostics)


if __name__ == '__main__':
    unittest.main()
