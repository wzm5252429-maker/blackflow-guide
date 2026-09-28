"""Independent battle-start modal boundaries, saved OCR and synthetic negatives."""
from dataclasses import replace
import unittest

import numpy as np

from blackflow_live.vision import OCRSpan, TemplateHit, VisionPipeline
from tests.test_live_vision import Templates, observer


class IndependentBattleConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720, 1280, 3), np.uint8)
        # Verbatim saved OCR from B development recording frame 2760, not live OCR.
        self.prompt = OCRSpan('应急干员机械师将在本次战斗后离开,是否要继续?',
                              .9550899267196655, (359,279,550,24))
        self.cancel = OCRSpan('取消', .8844771385192871, (336,469,56,33))
        self.confirm = OCRSpan('确认', .9999576210975647, (990,469,52,33))
        self.event = 'BlackFlow@Roguelike@CloseEvent.png'

    def analyze(self, spans, markers=()):
        hits = {name: [TemplateHit(name,.95,(939,467,42,42),1)] for name in markers}
        return observer(Templates(hits)).analyze(self.image, tuple(spans),
                                                frame_id='offline', captured_at=0.)

    def detect(self, spans):
        return VisionPipeline._battle_start_confirmation(self.image, tuple(spans))

    def test_real_warning_blocks_event_confirmation_without_checkbox_requirement(self):
        obs = self.analyze((self.prompt,self.cancel,self.confirm), (self.event,))
        self.assertEqual(obs.scene,'battle_start')
        self.assertEqual(obs.confidence,self.prompt.confidence)
        self.assertIn('battle_start_confirmation_requires_manual',obs.diagnostics)
        self.assertFalse(obs.actions)

    def test_operator_name_and_current_resolution_are_not_fixed(self):
        prompt = replace(self.prompt,text='应急干员预备干员将在本次战斗后离开，是否要继续？')
        for scale in (.5,1.,1.5):
            with self.subTest(scale=scale):
                self.image = np.zeros((int(720*scale),int(1280*scale),3),np.uint8)
                spans = [replace(s,bbox=tuple(v*scale for v in s.bbox))
                         for s in (prompt,self.cancel,self.confirm)]
                self.assertGreater(self.detect(spans),0)

    def test_warning_dominates_event_movement_map_and_recruitment_markers(self):
        markers = (self.event,'BlackFlow@Roguelike@MovePreviewEnter.png',
                   'BlackFlow@Roguelike@MapZoomIn.png',
                   'BlackFlow@Roguelike@ChooseOperConfirm.png')
        obs = self.analyze((self.prompt,self.cancel,self.confirm,
                            OCRSpan('出发前往',.99,(1120,550,100,30))),markers)
        self.assertEqual(obs.scene,'battle_start')
        self.assertFalse(obs.actions)
        self.assertFalse(obs.nodes)

    def test_ordinary_battle_narrative_retains_noncombat_event_confirmation(self):
        for text in ('经过一场战斗，你发现了遗失的物品。',
                     '本次战斗后离开树林，是否要继续？',
                     '应急干员想起往日战斗，是否要继续交谈？',
                     '招募干员需要消耗希望，是否要继续？'):
            with self.subTest(text=text):
                obs = self.analyze((replace(self.prompt,text=text),self.cancel,self.confirm),(self.event,))
                self.assertEqual(obs.scene,'event')
                self.assertTrue(any(a.label=='确认' for a in obs.actions))

    def test_scattered_fragments_are_not_joined_into_warning(self):
        fragments = (OCRSpan('应急干员',.99,(150,180,100,25)),
                     OCRSpan('本次战斗后离开',.99,(730,350,220,25)),
                     OCRSpan('是否要继续',.99,(400,240,140,25)))
        self.assertEqual(self.detect((*fragments,self.cancel,self.confirm)),0)
        self.assertEqual(self.detect((fragments[2],fragments[0],fragments[1],
                                      self.cancel,self.confirm)),0)

    def test_missing_or_wrong_button_layout_is_not_same_modal(self):
        cases = ((self.cancel,), (self.confirm,),
                 (replace(self.cancel,bbox=self.confirm.bbox),replace(self.confirm,bbox=self.cancel.bbox)),
                 (replace(self.cancel,bbox=(336,230,56,33)),self.confirm),
                 (self.cancel,replace(self.confirm,bbox=(990,570,52,33))),
                 (self.cancel,replace(self.confirm,text='确认购买')))
        for buttons in cases:
            with self.subTest(buttons=buttons):
                self.assertEqual(self.detect((self.prompt,*buttons)),0)

    def test_unreliable_or_off_panel_prompt_does_not_trigger(self):
        prompts = (replace(self.prompt,confidence=.89),replace(self.prompt,confidence=float('nan')),
                   replace(self.prompt,bbox=(10,279,550,24)),
                   replace(self.prompt,bbox=(359,100,550,24)),
                   replace(self.prompt,bbox=(359,500,550,24)),
                   replace(self.prompt,bbox=(359,279,0,24)),
                   replace(self.prompt,bbox=(359,float('inf'),550,24)))
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(self.detect((prompt,self.cancel,self.confirm)),0)

    def test_unreliable_or_invalid_button_does_not_trigger(self):
        for confirm in (replace(self.confirm,confidence=.79),
                        replace(self.confirm,bbox=(1250,469,52,33)),
                        replace(self.confirm,bbox=(990,469,52,-1)),
                        replace(self.confirm,bbox=(float('nan'),469,52,33))):
            with self.subTest(confirm=confirm):
                self.assertEqual(self.detect((self.prompt,self.cancel,confirm)),0)


if __name__ == '__main__':
    unittest.main()
