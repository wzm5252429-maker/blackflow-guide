"""Perception/grounding regressions. All input is synthetic or saved screenshots."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from blackflow_live.vision import (
    OCRSpan, TemplateHit, VisionPipeline, MaaTemplates, CorridorNet,
    extract_resources,
)


class EmptyOCR:
    def recognize(self,image): return ()


class Templates:
    node_specs=[]
    labels={}
    def __init__(self,hits=None): self.hits=hits or {}
    def match(self,image,name,**kwargs): return self.hits.get(name,[])


class EmptyCorridor:
    config={"decision":{"probability_threshold":.89}}
    def score(self,image,pairs): return [0.0]*len(pairs)


def observer(templates=None):
    return VisionPipeline(maa_root="missing",ocr=EmptyOCR(),templates=templates or Templates(),corridor=EmptyCorridor())


class LiveVisionTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((720,1280,3),np.uint8)

    def analyze(self,spans,templates=None):
        return observer(templates).analyze(self.image,spans,frame_id="frame-1",captured_at=123.0)

    def test_labeled_resources_only(self):
        spans=[OCRSpan("行动力：6",.98,(20,20,80,20)),OCRSpan("目标生命值4/5",.99,(20,40,100,20)),OCRSpan("源石锭25",.99,(20,60,100,20)),OCRSpan("收藏品：12",.99,(20,80,100,20))]
        self.assertEqual(extract_resources(spans),{"action_points":6,"hp":4,"max_hp":5,"gold":25,"relics":12})

    def test_costs_and_bare_numbers_do_not_become_current_resources(self):
        spans=[OCRSpan(text,1,(20,20,100,20)) for text in ("获得源石锭25","消耗希望3","行动力+5","25","构想3")]
        self.assertEqual(extract_resources(spans),{})

    def test_low_confidence_resource_is_unknown(self):
        self.assertEqual(extract_resources([OCRSpan("生命值1/4",.7,(1,1,10,10))]),{})

    def test_settings_with_confirm_word_do_not_create_actions(self):
        obs=self.analyze([OCRSpan("设置",.99,(20,20,60,20)),OCRSpan("确定",.99,(900,600,80,30))])
        self.assertEqual(obs.scene,"unknown")
        self.assertFalse(obs.actions)

    def test_battle_start_dominates_map_and_buttons(self):
        obs=self.analyze([OCRSpan("玻利瓦尔肤层",.99,(500,0,100,20)),OCRSpan("开始行动",.99,(900,600,100,30)),OCRSpan("确认",.99,(700,400,50,30))])
        self.assertEqual(obs.scene,"battle_start")
        self.assertFalse(obs.actions)

    def test_battle_template_yields_no_actions(self):
        name="BattleOfficiallyBegin.png"
        obs=self.analyze([OCRSpan("确认",.99,(700,400,50,30))],Templates({name:[TemplateHit(name,.97,(1150,10,60,60),1)]}))
        self.assertEqual(obs.scene,"battle")
        self.assertFalse(obs.actions)

    def test_generic_pass_never_confirms_first_ending(self):
        name="BlackFlow@Roguelike@GamePass.png"
        obs=self.analyze([],Templates({name:[TemplateHit(name,.95,(10,10,50,50),1)]}))
        self.assertEqual(obs.scene,"ending")
        self.assertFalse(obs.ending_first_confirmed)

    def test_recruitment_checkmark_does_not_misclassify_as_ending(self):
        ending='BlackFlow@Roguelike@GamePassTheEndConfirm.png'
        recruitment='BlackFlow@Roguelike@ChooseOperConfirm.png'
        templates=Templates({ending:[TemplateHit(ending,.92,(1114,667,29,31),1)],
                             recruitment:[TemplateHit(recruitment,.93,(1080,660,179,45),1)]})
        obs=self.analyze([OCRSpan('确认招募',.98,(1154,666,84,24)),
                          OCRSpan('放弃',.99,(978,667,41,23))],templates)
        self.assertEqual(obs.scene,'recruitment')
        self.assertFalse(obs.ending_first_confirmed)
        self.assertNotIn('ending',obs.metadata['markers'])
        # An independent result badge still takes priority over card controls.
        game_pass='BlackFlow@Roguelike@GamePass.png'
        templates.hits[game_pass]=[TemplateHit(game_pass,.98,(500,100,100,80),1)]
        result=self.analyze([OCRSpan('确认招募',.98,(1154,666,84,24))],templates)
        self.assertEqual(result.scene,'ending')

    def test_first_ending_requires_visible_result_evidence(self):
        obs=self.analyze([OCRSpan("探索完成",.99,(400,180,150,40)),OCRSpan("结局一达成",.99,(400,300,180,40))])
        self.assertEqual(obs.scene,"ending_complete")
        self.assertTrue(obs.ending_first_confirmed)
        self.assertGreaterEqual(obs.confidence,.95)
        self.assertFalse(obs.actions)

    def test_shared_checkmark_without_result_ui_never_proves_an_ending(self):
        name='BlackFlow@Roguelike@GamePassTheEndConfirm.png'
        templates=Templates({name:[TemplateHit(name,.99,(1114,667,29,31),1)]})
        for spans in ([],[OCRSpan('强制重启',.99,(400,200,100,30))]):
            obs=self.analyze(spans,templates)
            self.assertEqual(obs.scene,'unknown')
            self.assertFalse(obs.ending_first_confirmed)
            self.assertFalse(obs.actions)
            self.assertIn('confirmation_icon',obs.metadata['markers'])

    def test_cursor_obscured_recruitment_text_does_not_restore_false_ending(self):
        name='BlackFlow@Roguelike@GamePassTheEndConfirm.png'
        recruit='BlackFlow@Roguelike@ChooseOperConfirm.png'
        templates=Templates({name:[TemplateHit(name,.92,(1114,667,29,31),1)],
                            recruit:[TemplateHit(recruit,.93,(1080,660,179,45),1)]})
        obs=self.analyze([OCRSpan('确认招',.99,(1154,666,84,24))],templates)
        self.assertEqual(obs.scene,'recruitment')
        self.assertFalse(obs.ending_first_confirmed)

    def test_client_first_ending_name_matches_only_result_ui(self):
        spans=[OCRSpan("探索完成",.99,(400,180,150,40)),OCRSpan("强制重启",.98,(400,300,180,40))]
        self.assertTrue(self.analyze(spans).ending_first_confirmed)
        self.assertFalse(self.analyze(spans[1:]).ending_first_confirmed)

    def test_first_ending_prose_without_result_is_not_completion(self):
        obs=self.analyze([OCRSpan("结局一达成",.99,(400,300,180,40))])
        self.assertFalse(obs.ending_first_confirmed)

    def test_event_exact_choice_is_grounded_at_same_pixels(self):
        v=observer()
        v.choices={"测试选项":[{"id":"test_choice","title":"测试选项"}]}
        span=OCRSpan("测试选项",.97,(900,360,180,35))
        obs=v.analyze(self.image,[span],frame_id="f",captured_at=1)
        self.assertEqual(obs.scene,"event")
        self.assertEqual(len(obs.actions),1)
        self.assertEqual(obs.actions[0].bbox,span.bbox)
        self.assertEqual(obs.actions[0].metadata["choice_id"],"test_choice")

    def test_event_narrative_on_left_is_not_choice(self):
        v=observer();v.choices={"测试选项":[{"id":"test_choice"}]}
        obs=v.analyze(self.image,[OCRSpan("测试选项",.97,(100,360,180,35))],frame_id="f",captured_at=1)
        self.assertFalse(obs.actions)

    def test_disabled_choice_metadata_is_not_enabled(self):
        v=observer();v.choices={"测试选项":[{"id":"test_choice"}]}
        spans=[OCRSpan("测试选项",.97,(900,360,180,35)),OCRSpan("条件未满足",.98,(920,405,140,24))]
        obs=v.analyze(self.image,spans,frame_id="f",captured_at=1)
        self.assertFalse(obs.actions[0].enabled)

    def test_shop_purchase_price_must_be_explicit_near_button(self):
        spans=[OCRSpan("坎诺特",.99,(40,100,100,30)),OCRSpan("购买",.99,(950,500,70,30)),OCRSpan("售价：8源石锭",.98,(900,450,150,30))]
        obs=self.analyze(spans)
        self.assertEqual(obs.scene,"shop")
        purchase=next(a for a in obs.actions if a.label=="购买")
        self.assertEqual(purchase.metadata["price"],8)
        without=self.analyze(spans[:2])
        self.assertNotIn("price",without.actions[0].metadata)

    def test_unique_item_name_yields_preview_not_invented_purchase(self):
        v=observer();v.item_names={"测试道具":"测试道具"}
        obs=v.analyze(self.image,[OCRSpan("坎诺特",.99,(40,100,100,30)),OCRSpan("测试道具",.99,(450,350,120,25))],frame_id="f",captured_at=1)
        self.assertEqual(obs.actions[0].kind,"item_preview")
        self.assertTrue(obs.actions[0].metadata["preview_only"])
        self.assertNotIn("price",obs.actions[0].metadata)
        self.assertNotIn("inventory_complete",obs.metadata)

    def test_arbitrary_image_dimensions_preserve_pixels(self):
        v=observer();v.choices={"测试选项":[{"id":"test_choice"}]}
        image=np.zeros((1000,2400,3),np.uint8)
        bbox=(1850,470,220,42)
        obs=v.analyze(image,[OCRSpan("测试选项",.99,bbox)],frame_id="wide",captured_at=1)
        self.assertEqual(obs.actions[0].bbox,bbox)
        self.assertEqual(obs.metadata["image_width"],2400)

    def test_frame_id_and_time_are_carried_without_stale_file_fallback(self):
        obs=observer().recognize(self.image,frame_id="newframe",captured_at=300)
        self.assertEqual((obs.frame_id,obs.captured_at),("newframe",300))
        self.assertEqual(obs.nodes,())
        self.assertEqual(obs.resources,{})

    def test_real_template_matching_at_translated_non_16_9_dimensions(self):
        import cv2
        rng=np.random.default_rng(4)
        templ=rng.integers(0,255,size=(20,30,3),dtype=np.uint8)
        image=np.zeros((800,1400,3),np.uint8)
        image[433:453,1221:1251]=templ
        matcher=MaaTemplates.__new__(MaaTemplates)
        matcher._cache={"fixture.png":templ}
        hits=matcher.match(image,"fixture.png",threshold=.95,scales=[1])
        self.assertEqual(hits[0].bbox,(1221,433,30,20))

    def test_corridor_pixels_use_manifest_rgb_normalization(self):
        captured={}
        class Input:
            name="input"
        class Session:
            def get_inputs(self): return [Input()]
            def run(self,_outputs,feeds):
                captured["input"]=feeds["input"]
                return [np.array([3.0],np.float32)]
        net=CorridorNet.__new__(CorridorNet)
        net.session=Session()
        net.config={"corridor":{"endpoint_margin_ratio":.16,"width_ratio_of_edge_length":.3,"minimum_width_pixels":16,"maximum_width_pixels":64},"preprocessing":{"scale":1/255,"mean":[.5,.5,.5],"std":[.25,.25,.25]},"decision":{"temperature":1}}
        image=np.full((100,200,3),(0,127,255),np.uint8)
        result=net.score(image,[((30,50),(170,50))])
        self.assertEqual(captured["input"].shape,(1,3,40,160))
        self.assertAlmostEqual(float(captured["input"][0,0,20,80]),2)
        self.assertAlmostEqual(float(captured["input"][0,2,20,80]),-2)
        self.assertGreater(result[0],.95)


if __name__=="__main__": unittest.main()
