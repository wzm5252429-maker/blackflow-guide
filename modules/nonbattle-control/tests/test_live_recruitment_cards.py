"""Offline real-pixel recruitment replay; no native capture or input."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from blackflow_live.models import LiveObservation
from blackflow_live.policy import ObservedMenuMetadataAdapter, action_is_safe
from blackflow_live.recruitment_cards import detect_recruitment_cards, _displayed_hope
from blackflow_live.vision import OCRSpan, _merge_operator_names


FIXTURES = Path(__file__).parent/'fixtures/live_recruitment_cards'
ROOT = Path(__file__).resolve().parents[1]


class RecordedOCR:
    """Only replay real crop readings when the complete pixel hash agrees."""
    def __init__(self, records):
        self.records = records
        self.unknown_crops = 0

    def recognize_crop(self, image):
        record = self.records.get(sha256(image.tobytes()).hexdigest())
        if record is None:
            self.unknown_crops += 1
            return '',0.
        assert list(image.shape) == record['shape']
        return record['text'],record['confidence']


class RecruitmentCardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
        snapshot = json.loads((ROOT/'data/evidence/rogue6_observed_operator_catalog_v1.json').read_text(encoding='utf-8'))
        cls.operators = _merge_operator_names({},snapshot['ordinary_and_exclusive_characters'])
        from blackflow_rl.catalog import load_catalog
        cls.adapter = ObservedMenuMetadataAdapter(load_catalog())

    def frame(self, second):
        case = next(c for c in self.manifest['cases'] if c['second']==second)
        path = FIXTURES/case['image']
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),case['sha256'])
        original = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        image = cv2.resize(original,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
        spans = [OCRSpan(s['text'],s['confidence'],tuple(s['bbox'])) for s in case['spans']]
        return image,spans,RecordedOCR(case['crop_ocr'])

    def detect(self, image, spans, ocr):
        return detect_recruitment_cards(image,spans,ocr=ocr,operators=self.operators)

    @staticmethod
    def confirmations(result):
        return [a for a in result.actions if a.metadata.get('selection_stage')=='operator_confirm']

    def test_free_selected_card_completes_the_real_confirmation_contract(self):
        image,spans,ocr = self.frame(8)
        result = self.detect(image,spans,ocr)
        self.assertEqual(ocr.unknown_crops,0)
        self.assertEqual(result.resources,{'hope':0})
        actions = self.confirmations(result)
        self.assertEqual(len(actions),1)
        action = actions[0]
        self.assertEqual(action.label,'确认招募')
        self.assertEqual(action.metadata['operator_name'],'地灵')
        self.assertEqual(action.metadata['operator_id'],'char_183_skgoat')
        self.assertEqual(action.metadata['selected_operator_id'],'char_183_skgoat')
        self.assertEqual(action.metadata['resource_costs'],{'hope':0})
        self.assertTrue(action.metadata['button_enabled_observed'])
        self.assertFalse(action.metadata['preview_only'])
        self.assertEqual(action.bbox,next(s.bbox for s in spans if s.text=='确认招募'))
        self.assertTrue(result.evidence['hope_display']['available_balance_verified'])
        # The integration supplies same-frame provenance; the policy adapter
        # independently verifies identity before its real safety mask runs.
        action = replace(action,metadata={**action.metadata,'source_frame_id':'fixture-8'})
        action = replace(action,metadata=self.adapter.adapt(action))
        obs = LiveObservation('fixture-8',0,'recruitment',result.confidence,(action,),
                              resources=result.resources,metadata={'image_width':1280,'image_height':720})
        self.assertTrue(action_is_safe(action,obs))
        self.assertFalse(action_is_safe(action,replace(obs,frame_id='new-frame')))
        self.assertFalse(action_is_safe(action,replace(obs,resources={})))

    def test_unselected_screen_provides_real_card_previews_and_no_confirmation(self):
        result = self.detect(*self.frame(10))
        self.assertEqual({a.label for a in result.actions},{'豆苗','讯使','芬','红豆'})
        self.assertFalse(self.confirmations(result))
        self.assertFalse(result.resources)
        for action in result.actions:
            self.assertEqual(action.kind,'operator_preview')
            self.assertEqual(action.metadata['operation'],'event')
            self.assertTrue(action.metadata['preview_only'])
            self.assertNotIn('resource_costs',action.metadata)
            self.assertNotIn('formal_operator_ids',action.metadata)

    def test_paid_selected_card_keeps_displayed_hope_distinct_from_available_balance(self):
        result = self.detect(*self.frame(4))
        self.assertEqual(result.evidence['hope_display']['displayed_hope'],0)
        self.assertFalse(result.evidence['hope_display']['available_balance_verified'])
        self.assertFalse(result.resources)
        self.assertFalse(self.confirmations(result))
        selected = [c for c in result.evidence['cards'] if c['highlighted']]
        self.assertEqual([c['hope_cost'] for c in selected],[6])
        self.assertIn('recruitment_hope_display_may_be_post_selection',result.diagnostics)

    def test_selected_card_is_not_offered_as_a_redundant_preview(self):
        result = self.detect(*self.frame(8))
        self.assertFalse([a for a in result.actions if a.kind=='operator_preview'
                          and a.metadata['operator_id']=='char_183_skgoat'])

    def test_missing_or_conflicting_left_detail_cannot_confirm(self):
        for mode in ('missing','conflicting','placeholder','duplicate'):
            with self.subTest(mode=mode):
                image,spans,ocr = self.frame(8)
                detail = next(s for s in spans if s.text=='地灵' and s.bbox[0]<100)
                spans.remove(detail)
                if mode=='conflicting':
                    spans.append(replace(detail,text='红豆'))
                elif mode=='placeholder':
                    spans.extend([detail,OCRSpan('轻触右侧干员以查看详情',.99,(62,168,193,19))])
                elif mode=='duplicate':
                    spans.extend([detail,detail])
                result = self.detect(image,spans,ocr)
                self.assertFalse(self.confirmations(result))
                self.assertFalse(result.resources)

    def test_same_left_name_without_matching_complete_card_name_is_insufficient(self):
        image,spans,ocr = self.frame(8)
        spans = [s for s in spans if not (s.text=='地灵' and s.bbox[0]>500)]
        result = self.detect(image,spans,ocr)
        self.assertFalse(self.confirmations(result))

    def test_prefix_names_and_duplicate_card_identities_are_not_guessed(self):
        for mode in ('prefix','duplicate'):
            image,spans,ocr = self.frame(8)
            name = next(s for s in spans if s.text=='地灵' and s.bbox[0]>500)
            if mode=='prefix':
                spans[spans.index(name)] = replace(name,text='地',confidence=.99)
            else:
                spans.append(replace(name,text='红豆',confidence=.99))
            result = self.detect(image,spans,ocr)
            self.assertFalse(self.confirmations(result))

    def test_greyed_confirmation_button_is_not_treated_as_enabled(self):
        image,spans,ocr = self.frame(8)
        image[650:710,1070:1250] = cv2.cvtColor(cv2.cvtColor(image[650:710,1070:1250],cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        result = self.detect(image,spans,ocr)
        self.assertFalse(result.evidence['button_enabled_observed'])
        self.assertFalse(self.confirmations(result))

    def test_name_and_blue_button_without_blue_card_cannot_confirm(self):
        image,spans,ocr = self.frame(8)
        image[238:348,371:683] = cv2.cvtColor(cv2.cvtColor(image[238:348,371:683],cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        result = self.detect(image,spans,ocr)
        self.assertFalse(self.confirmations(result))

    def test_fee_or_currency_icon_occlusion_cannot_borrow_a_neighbor_zero(self):
        for rectangle in ((588,289,30,27),(554,287,28,27)):
            image,spans,ocr = self.frame(8)
            x,y,w,h = rectangle
            image[y:y+h,x:x+w] = 0
            spans = [s for s in spans if not (580<s.bbox[0]<620 and 280<s.bbox[1]<315)]
            result = self.detect(image,spans,ocr)
            self.assertFalse(self.confirmations(result))
            self.assertFalse(result.resources)

    def test_unknown_current_cell_does_not_read_maximum_as_the_balance(self):
        image,spans,ocr = self.frame(8)
        image[15:42,489:522] = 0
        result = self.detect(image,spans,ocr)
        self.assertIsNone(result.evidence['hope_display']['displayed_hope'])
        self.assertFalse(result.resources)
        self.assertFalse(self.confirmations(result))

    def test_real_six_and_animation_two_are_not_matched_as_zero(self):
        image,_,_ = self.frame(3)
        self.assertIsNone(_displayed_hope(image,1)[0])
        record = self.manifest['display_counterexamples'][0]
        path = FIXTURES/record['image']
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),record['sha256'])
        original = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        image = cv2.resize(original,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
        self.assertIsNone(_displayed_hope(image,1)[0])

    def test_occluding_footer_without_readable_cards_still_blocks_background(self):
        image,spans,ocr = self.frame(8)
        spans = [s for s in spans if s.text in ('确认招募','放弃')]
        image[100:640,350:] = 220
        result = self.detect(image,spans,ocr)
        self.assertIsNotNone(result)
        self.assertEqual(result.evidence['kind'],'regular_operator_cards')
        self.assertFalse(result.actions)

    def test_complete_card_band_is_reread_without_relying_on_a_global_name_prefix(self):
        image,spans,ocr = self.frame(3)
        spans = [s for s in spans if s.text in ('确认招募','放弃')]
        result = self.detect(image,spans,ocr)
        self.assertEqual([a.label for a in result.actions],['新约能天使'])
        self.assertEqual(ocr.unknown_crops,0)
        self.assertGreater(result.actions[0].confidence,.99)
        self.assertEqual(result.actions[0].kind,'operator_preview')

    def test_full_card_band_disagreement_or_incomplete_text_does_not_create_identity(self):
        for mode in ('disagree','incomplete'):
            image,spans,ocr = self.frame(3)
            spans = [s for s in spans if s.text in ('确认招募','放弃')]
            ocr.records = {key:dict(value) for key,value in ocr.records.items()}
            matching = [(key,record) for key,record in ocr.records.items()
                        if record['text']=='新约能天使' and record['confidence']>.99]
            self.assertGreaterEqual(len(matching),2)
            if mode=='incomplete':
                for _,record in matching:
                    record['text']='新约能'
            else:
                matching[0][1]['text']='红豆'
            result = self.detect(image,spans,ocr)
            self.assertFalse(result.actions)

    def test_footer_pair_is_required_and_ambiguous_footer_waits(self):
        image,spans,ocr = self.frame(8)
        self.assertIsNone(self.detect(image,[s for s in spans if s.text!='放弃'],ocr))
        spans.append(next(s for s in spans if s.text=='确认招募'))
        result = self.detect(image,spans,ocr)
        self.assertFalse(result.actions)
        self.assertIn('recruitment_footer_ambiguous',result.diagnostics)

    def test_wider_offset_canvas_keeps_pixel_bound_action_coordinates(self):
        image,spans,ocr = self.frame(8)
        canvas = np.full((750,1580,3),220,np.uint8)
        canvas[20:740,150:1430] = image
        shifted = [replace(s,bbox=(s.bbox[0]+150,s.bbox[1]+20,*s.bbox[2:])) for s in spans]
        result = self.detect(canvas,shifted,ocr)
        actions = self.confirmations(result)
        self.assertEqual(len(actions),1)
        self.assertEqual(actions[0].bbox,(1301,689,80,24))
        self.assertEqual(result.resources,{'hope':0})

    def test_recognition_scale_loss_stays_unknown_instead_of_using_defaults(self):
        # Runtime normally normalizes recognition to height 720. Deliberately
        # degraded raw scales must not reuse readings from the reference crop.
        for replay in self.manifest['scale_replays']:
            case = next(c for c in self.manifest['cases'] if c['image']==replay['source_image'])
            source = cv2.imdecode(np.fromfile(str(FIXTURES/case['image']),np.uint8),cv2.IMREAD_COLOR)
            scale = replay['scale_from_ocr_reference']
            image = cv2.resize(source,None,fx=2*scale,fy=2*scale,interpolation=cv2.INTER_LANCZOS4)
            spans = [OCRSpan(s['text'],s['confidence'],tuple(v*scale for v in s['bbox'])) for s in case['spans']]
            result = self.detect(image,spans,RecordedOCR(replay['crop_ocr']))
            self.assertFalse(self.confirmations(result))
            self.assertFalse(result.resources)


if __name__=='__main__':
    unittest.main()
