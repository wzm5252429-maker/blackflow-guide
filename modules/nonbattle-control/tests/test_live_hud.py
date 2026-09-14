"""Label-bound HUD numbers; pixel regression uses the user's supplied image."""
from pathlib import Path
import json
import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.vision import DEFAULT_MAA_ROOT, OCRSpan, VisionPipeline, _read_image


FIXTURES=Path(__file__).parent/'fixtures/live_hud'


class NoTemplates:
    def match(self,*args,**kwargs):
        return []


class HudAssociationTests(unittest.TestCase):
    def pipeline(self):
        pipeline=VisionPipeline.__new__(VisionPipeline)
        pipeline.templates=NoTemplates()
        pipeline.ocr=Mock()
        pipeline.ocr.recognize_crop.return_value=('',0)
        return pipeline

    def test_existing_fraction_is_preserved_without_bad_recrop(self):
        pipeline=self.pipeline()
        image=np.zeros((720,1246,3),np.uint8)
        spans=[OCRSpan('目标生命值',.945,(153,11,70,15)),
               OCRSpan('4/4',.986,(155,31,31,19)),
               OCRSpan('2',.999,(219,31,29,16))]
        self.assertEqual(pipeline._hud_resources(image,spans),{'hp':4,'max_hp':4})
        pipeline.ocr.recognize_crop.assert_not_called()

    def test_fraction_must_be_under_the_actual_label(self):
        pipeline=self.pipeline()
        image=np.zeros((720,1246,3),np.uint8)
        for box in ((155,400,31,19),(300,31,31,19)):
            spans=[OCRSpan('目标生命值',.99,(153,11,70,15)),OCRSpan('4/4',.99,box)]
            self.assertNotIn('hp',pipeline._hud_resources(image,spans))

    def test_ambiguous_or_impossible_fraction_stays_unknown(self):
        pipeline=self.pipeline()
        image=np.zeros((720,1246,3),np.uint8)
        label=OCRSpan('目标生命值',.99,(153,11,70,15))
        for values in (['5/4'],['3/4','4/4']):
            spans=[label]+[OCRSpan(value,.99,(155,31,31,19)) for value in values]
            self.assertNotIn('hp',pipeline._hud_resources(image,spans))

    def test_weak_fraction_is_not_promoted_by_a_strong_label(self):
        pipeline=self.pipeline()
        spans=[OCRSpan('目标生命值',.99,(153,11,70,15)),OCRSpan('4/4',.6,(155,31,31,19))]
        self.assertNotIn('hp',pipeline._hud_resources(np.zeros((720,1246,3),np.uint8),spans))

    def test_conflicting_numeric_preprocessing_does_not_choose_first(self):
        pipeline=self.pipeline()
        pipeline.ocr.recognize_crop.side_effect=[('1',.99),('7',.99)]
        self.assertIsNone(pipeline._numeric_crop(np.zeros((40,50,3),np.uint8),(2,2,30,30),contrast=True))

    def test_blank_icon_does_not_imply_zero_or_one_relic(self):
        pipeline=self.pipeline()
        spans=[OCRSpan('收藏品',.999,(128,688,58,23))]
        self.assertNotIn('relics',pipeline._hud_resources(np.zeros((720,1246,3),np.uint8),spans))


@unittest.skipUnless((DEFAULT_MAA_ROOT/'resource/PaddleOCR/rec/inference.onnx').is_file(),
                     'Installed MAA OCR assets required for real-pixel integration test')
class SavedHudPixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline=VisionPipeline()

    def test_supplied_fullscreen_hud_all_seven_fields(self):
        for name in ('fullscreen_hud','native_fullscreen_hud'):
            with self.subTest(source=name):
                image=_read_image(FIXTURES/(name+'.png'))
                expected=json.loads((FIXTURES/(name+'.json')).read_text(encoding='utf-8'))['expected']
                spans=self.pipeline.ocr.recognize(image)
                self.assertEqual(self.pipeline._hud_resources(image,spans),expected)


if __name__=='__main__':
    unittest.main()
