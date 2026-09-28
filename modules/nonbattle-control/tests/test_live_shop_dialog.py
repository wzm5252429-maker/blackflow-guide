"""Shop modal grounding: same-dialog facts, exact buttons and no hidden shelf."""
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.shop_vision import detect_shop_dialog
from blackflow_live.vision import DEFAULT_MAA_ROOT, OCRSpan, VisionPipeline, _read_image


ITEMS = {'浪花': '浪花', '种子': '种子'}
FIXTURES = Path(__file__).parent/'fixtures/live_shop_dialog'


def spans(operation='sell', button_confidence=.877):
    return (
        OCRSpan('浪花', .999, (508,184,41,24)),
        OCRSpan('是否以' if operation == 'sell' else '是否消耗', .999, (651,419,60,23)),
        OCRSpan('18的价格出售浪花?' if operation == 'sell' else '18购买浪花?', .889, (738,420,168,21)),
        OCRSpan('算了', .999, (605,468,40,24)),
        OCRSpan('确认出售' if operation == 'sell' else '确认购买', button_confidence, (912,469,75,22)),
    )


def observer(ocr):
    return VisionPipeline(maa_root='missing', ocr=ocr,
                          templates=SimpleNamespace(match=lambda *args, **kwargs: [], labels={}, node_specs=[]),
                          corridor=SimpleNamespace(score=lambda *args: [], config={'decision': {'probability_threshold': .9}}))


class ShopDialogTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720,1280,3),np.uint8)

    def reader(self, operation='sell', *, weak_button=True):
        ocr = Mock()
        readings = [('确认出售' if operation == 'sell' else '确认购买', .98)]*2 if weak_button else []
        question = '是否以18的价格出售浪花?' if operation == 'sell' else '是否消耗18购买浪花?'
        ocr.recognize_crop.side_effect = readings + [(question,.98)]*2
        return ocr

    def test_sale_uses_shown_amount_and_name_without_inventing_balance(self):
        result = detect_shop_dialog(self.image, spans(), ocr=self.reader(), item_names=ITEMS)
        self.assertIsNotNone(result)
        sale,cancel = result.grounded_actions({})
        self.assertEqual(sale.metadata['operation'], 'sell')
        self.assertEqual(sale.metadata['item_name'], '浪花')
        self.assertEqual(sale.metadata['price'], 18)
        self.assertEqual(sale.metadata['resource_delta'], {'gold':18})
        self.assertEqual(cancel.metadata['operation'], 'event_advance')
        self.assertNotIn('ends_node', cancel.metadata)

    def test_purchase_requires_explicit_price_and_current_observed_gold(self):
        result = detect_shop_dialog(self.image, spans('purchase', .99),
                                    ocr=self.reader('purchase',weak_button=False), item_names=ITEMS)
        for resources in ({}, {'gold':None}, {'gold':17}, {'gold':float('nan')}):
            self.assertEqual([a.label for a in result.grounded_actions(resources)], ['算了'])
        purchase,cancel = result.grounded_actions({'gold':18})
        self.assertEqual(purchase.metadata['resource_delta'], {'gold':-18})
        self.assertEqual(purchase.metadata['operation'], 'purchase')

    def test_conflicting_title_or_question_amount_suppresses_confirmation(self):
        conflicting_title = spans()+(OCRSpan('种子',.999,(680,185,42,24)),)
        result = detect_shop_dialog(self.image, conflicting_title, ocr=self.reader(),item_names=ITEMS)
        self.assertEqual([a.label for a in result.actions], ['算了'])
        ocr = self.reader()
        ocr.recognize_crop.side_effect = [('确认出售',.98)]*2 + [
            ('是否以18的价格出售浪花?',.99),('是否以8的价格出售浪花?',.99)]
        result = detect_shop_dialog(self.image, spans(), ocr=ocr,item_names=ITEMS)
        self.assertEqual([a.label for a in result.actions], ['算了'])

    def test_incomplete_question_does_not_borrow_shelf_price(self):
        ocr = Mock()
        ocr.recognize_crop.return_value = ('是否消耗',.99)
        background = OCRSpan('价格18',.999,(600,220,70,30))
        result = detect_shop_dialog(self.image, spans('purchase',.99)+(background,),ocr=ocr,item_names=ITEMS)
        self.assertEqual([a.label for a in result.actions], ['算了'])
        self.assertNotIn('price',result.evidence)

    def test_weak_button_requires_agreeing_local_recognition(self):
        ocr = self.reader()
        ocr.recognize_crop.side_effect = [('确认出售',.98),('确认出他',.99)] + [('是否以18的价格出售浪花?',.99)]*2
        result = detect_shop_dialog(self.image, spans(),ocr=ocr,item_names=ITEMS)
        self.assertEqual([a.label for a in result.actions], ['算了'])

    def test_unrelated_cancel_cannot_make_dialog(self):
        source = list(spans())
        value = source[3]
        source[3] = OCRSpan(value.text,value.confidence,(100,100,40,24))
        self.assertIsNone(detect_shop_dialog(self.image,source,ocr=self.reader(),item_names=ITEMS))

    def test_verified_footer_blocks_background_when_question_is_missing_weak_or_ambiguous(self):
        for defect in ('missing', 'weak', 'elsewhere', 'ambiguous'):
            with self.subTest(defect=defect):
                source = list(spans('purchase', .99))
                question = source[1]
                if defect == 'missing':
                    source.pop(1)
                elif defect == 'weak':
                    source[1] = OCRSpan(question.text,.84,question.bbox)
                elif defect == 'elsewhere':
                    source[1] = OCRSpan(question.text,.99,(30,100,60,23))
                else:
                    source.append(OCRSpan('是否消耗',.99,(750,405,60,23)))
                source.extend((OCRSpan('卡德霍之颅',.99,(600,10,100,22)),
                               OCRSpan('离开',.99,(1180,560,65,35)),
                               OCRSpan('购买',.99,(900,260,70,25))))
                ocr = Mock()
                pipeline = observer(ocr)
                pipeline.item_names = ITEMS
                pipeline._map = Mock(side_effect=AssertionError('occluded map must not run'))
                obs = pipeline.analyze(self.image,source,frame_id='saved-test',captured_at=0)
                self.assertEqual(obs.scene,'shop')
                self.assertEqual([a.label for a in obs.actions],['算了'])
                self.assertEqual(obs.actions[0].metadata['operation'],'event_advance')
                self.assertIn('shop_confirmation_question_missing_or_ambiguous',obs.diagnostics)
                self.assertFalse(obs.nodes)
                pipeline._map.assert_not_called()
                ocr.recognize_crop.assert_not_called()

    def test_modal_overrides_map_and_suppresses_background_buttons(self):
        pipeline = observer(self.reader())
        pipeline.item_names = ITEMS
        source = spans()+(OCRSpan('卡德霍之颅',.99,(600,10,100,22)),
                          OCRSpan('离开',.99,(1180,560,65,35)), OCRSpan('种子',.99,(40,170,40,25)))
        obs = pipeline.analyze(self.image,source,frame_id='saved-test',captured_at=0)
        self.assertEqual(obs.scene,'shop')
        self.assertEqual([a.label for a in obs.actions],['确认出售','算了'])
        self.assertFalse(obs.nodes)
        self.assertEqual(obs.resources,{})
        self.assertTrue(all(a.metadata['source_frame_id']=='saved-test' for a in obs.actions))

    def test_ambiguous_footer_suppresses_background_without_guessing_a_trade(self):
        for defect in ('overlapping_confirmations', 'multiple_cancels'):
            with self.subTest(defect=defect):
                source = spans('purchase', .99)
                if defect == 'overlapping_confirmations':
                    source += (OCRSpan('确认出售',.99,(914,470,75,22)),)
                else:
                    source += (OCRSpan('算了',.99,(710,469,40,24)),)
                source += (OCRSpan('卡德霍之颅',.99,(600,10,100,22)),
                           OCRSpan('离开',.99,(1180,560,65,35)))
                ocr = Mock()
                pipeline = observer(ocr)
                pipeline.item_names = ITEMS
                pipeline._map = Mock(side_effect=AssertionError('occluded map must not run'))
                obs = pipeline.analyze(self.image,source,frame_id='saved-test',captured_at=0)
                self.assertEqual(obs.scene,'shop')
                self.assertEqual([a.label for a in obs.actions],
                                 ['算了'] if defect == 'overlapping_confirmations' else [])
                self.assertTrue(all(a.metadata['operation']=='event_advance' for a in obs.actions))
                self.assertIn('shop_confirmation_footer_ambiguous',obs.diagnostics)
                self.assertFalse(obs.nodes)
                pipeline._map.assert_not_called()
                ocr.recognize_crop.assert_not_called()

    def test_battle_remains_higher_priority_than_shop_confirmation(self):
        ocr = self.reader()
        pipeline = observer(ocr)
        pipeline.item_names = ITEMS
        obs = pipeline.analyze(self.image,spans()+(OCRSpan('开始行动',.99,(900,650,100,40)),),
                               frame_id='saved-test',captured_at=0)
        self.assertEqual(obs.scene,'battle_start')
        self.assertFalse(obs.actions)
        ocr.recognize_crop.assert_not_called()


@unittest.skipUnless((DEFAULT_MAA_ROOT/'resource/PaddleOCR/rec/inference.onnx').is_file(),
                     'Installed MAA OCR assets required for real-pixel integration test')
class SavedShopDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from blackflow_live.vision import MaaPaddleOCR
        cls.ocr = MaaPaddleOCR()

    def test_clear_sale_and_transition_purchase_from_existing_video(self):
        from PIL import Image
        manifest = json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['current_user_native_capture'])
        for case in manifest['cases']:
            with self.subTest(source=case['image']):
                source = _read_image(FIXTURES/case['image'])
                image = np.asarray(Image.fromarray(source).resize((1280,720),Image.Resampling.LANCZOS))
                current_spans = self.ocr.recognize(image)
                result = detect_shop_dialog(image,current_spans,ocr=self.ocr,item_names=ITEMS)
                self.assertIsNotNone(result)
                if case['expected'] == 'sell':
                    self.assertEqual([a.label for a in result.actions],['确认出售','算了'])
                    self.assertEqual(result.actions[0].metadata['price'],18)
                    self.assertEqual(result.actions[0].metadata['item_name'],'浪花')
                else:
                    self.assertEqual([a.label for a in result.actions],['算了'])
                    self.assertNotIn('price',result.evidence)


if __name__ == '__main__':
    unittest.main()
