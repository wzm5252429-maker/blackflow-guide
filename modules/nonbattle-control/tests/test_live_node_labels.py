"""Clipped map titles are re-read from pixels, never completed from prefixes."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.vision import DEFAULT_MAA_ROOT, MaaPaddleOCR, OCRSpan, VisionPipeline, _read_image


LABELS = {'未知的凶戾': 'hide_battle', '未知的诡秘': 'hide_invisible', '险路尽头': 'final'}
FIXTURES = Path(__file__).parent/'fixtures/live_nodes'


def pipeline(ocr):
    result = object.__new__(VisionPipeline)
    result.templates = SimpleNamespace(labels=LABELS)
    result.ocr = ocr
    return result


class NodeLabelAssociationTests(unittest.TestCase):
    def test_complete_label_is_preserved_without_another_ocr_call(self):
        ocr = Mock()
        span = OCRSpan('未知的凶戾', .99, (20, 20, 65, 13))
        self.assertIs(pipeline(ocr)._node_label_span(np.zeros((80, 160, 3), np.uint8), span), span)
        ocr.recognize_crop.assert_not_called()

    def test_retry_requires_same_image_complete_title_and_preserves_center(self):
        image = np.arange(80*160*3, dtype=np.uint8).reshape(80, 160, 3)
        ocr = Mock()
        ocr.recognize_crop.return_value = ('未知的凶戾', .99)
        span = OCRSpan('未知的凶', .905, (20, 20, 66, 13))
        result = pipeline(ocr)._node_label_span(image, span)
        self.assertEqual(result.text, '未知的凶戾')
        self.assertEqual(result.bbox, span.bbox)
        np.testing.assert_array_equal(ocr.recognize_crop.call_args.args[0], image[17:36, 16:90])

    def test_partial_ambiguous_wrong_or_weak_retry_stays_unknown(self):
        for text, confidence in (('未知的', .999), ('未知的凶', .999), ('险路尽头', .999),
                                 ('未知的凶戾', .89), ('未知的凶戾', float('nan'))):
            with self.subTest(text=text, confidence=confidence):
                ocr = Mock()
                ocr.recognize_crop.return_value = (text, confidence)
                result = pipeline(ocr)._node_label_span(np.zeros((80, 160, 3), np.uint8),
                                                       OCRSpan('未知的', .99, (20, 20, 41, 15)))
                self.assertIsNone(result)

    def test_unrelated_text_short_prefix_and_low_confidence_do_not_trigger_retry(self):
        for text, confidence in (('未知', .99), ('设置', .99), ('未知的凶', .7)):
            with self.subTest(text=text):
                ocr = Mock()
                self.assertIsNone(pipeline(ocr)._node_label_span(np.zeros((80, 160, 3), np.uint8),
                                                               OCRSpan(text, confidence, (20, 20, 41, 15))))
                ocr.recognize_crop.assert_not_called()


@unittest.skipUnless((DEFAULT_MAA_ROOT/'resource/PaddleOCR/rec/inference.onnx').is_file(),
                     'Installed MAA OCR assets required for real-pixel integration test')
class SavedNodeLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = pipeline(MaaPaddleOCR())

    def test_three_frame_titles_recover_but_cursor_occlusion_does_not_invent_node(self):
        cases = json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))['cases']
        for case in cases:
            with self.subTest(source=case['image']):
                image = _read_image(FIXTURES/case['image'])
                span = OCRSpan(**case['span'])
                result = self.pipeline._node_label_span(image, span)
                if case['expected'] is None:
                    self.assertIsNone(result)
                else:
                    self.assertIsNotNone(result)
                    self.assertEqual(result.text, case['expected'])
                    self.assertGreaterEqual(result.confidence, .90)


if __name__ == '__main__':
    unittest.main()
