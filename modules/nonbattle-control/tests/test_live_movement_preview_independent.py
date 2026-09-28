"""Independent movement panel boundaries; synthetic spans/hits, no live input."""
from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.vision import OCRSpan, TemplateHit
from tests.test_live_vision import Templates, observer

ENTRY = 'BlackFlow@Roguelike@MovePreviewEnter.png'
MAP = 'BlackFlow@Roguelike@MapZoomIn.png'


class LocalTemplates(Templates):
    def __init__(self, local=(), global_hits=None):
        super().__init__(global_hits)
        self.local = list(local)
        self.local_calls = []
    def template(self, name):
        return np.zeros((30,100,3),np.uint8) if name == ENTRY else None
    def match(self, image, name, **kwargs):
        if name == ENTRY and 'scales' in kwargs:
            self.local_calls.append((image.copy(),kwargs))
            return self.local
        return super().match(image,name,**kwargs)


class IndependentMovementPreviewTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((720,1280,3),np.uint8)
        self.span=OCRSpan('出发前往',.97,(1100,550,100,30))
        self.local_hit=TemplateHit(ENTRY,.94,(18,18,100,30),1.)
        self.map_hit=TemplateHit(MAP,.99,(10,530,40,40),1.)

    def analyze(self, spans, templates):
        p=observer(templates)
        p._map=Mock(side_effect=AssertionError('Detail overlay must not click the background map'))
        result=p.analyze(self.image,spans,frame_id='test-frame',captured_at=100.)
        return result,p

    def test_text_alone_blocks_background_map_and_generic_confirmation(self):
        templates=LocalTemplates(global_hits={MAP:[self.map_hit]})
        obs,p=self.analyze((self.span,OCRSpan('确认',.99,(1150,630,50,25))),templates)
        self.assertEqual(obs.scene,'movement_preview')
        self.assertFalse(obs.actions)
        self.assertFalse(obs.nodes)
        p._map.assert_not_called()

    def test_unavailable_text_suppresses_even_a_global_entry_match(self):
        templates=LocalTemplates((self.local_hit,),{ENTRY:[TemplateHit(ENTRY,.99,self.span.bbox,1)],MAP:[self.map_hit]})
        obs,_=self.analyze((replace(self.span,text='无法抵达'),),templates)
        self.assertEqual(obs.scene,'movement_preview')
        self.assertIn('movement_destination_unavailable',obs.diagnostics)
        self.assertFalse(obs.actions)
        self.assertFalse(templates.local_calls)

    def test_duplicate_hints_never_choose_one_by_confidence(self):
        for global_entry in (False,True):
            with self.subTest(global_entry=global_entry):
                hits={MAP:[self.map_hit]}
                if global_entry:hits[ENTRY]=[TemplateHit(ENTRY,.99,self.span.bbox,1)]
                templates=LocalTemplates((self.local_hit,),hits)
                second=replace(self.span,confidence=.81,bbox=(1120,600,100,30))
                obs,_=self.analyze((self.span,second),templates)
                self.assertEqual(obs.scene,'movement_preview')
                self.assertIn('ambiguous_movement_entry',obs.diagnostics)
                self.assertFalse(obs.actions)
                self.assertFalse(templates.local_calls)

    def test_disabled_and_enabled_hints_together_keep_panel_blocked(self):
        templates=LocalTemplates((self.local_hit,))
        obs,_=self.analyze((self.span,replace(self.span,text='无法抵达',bbox=(1120,600,100,30))),templates)
        self.assertFalse(obs.actions)
        self.assertIn('movement_destination_unavailable',obs.diagnostics)

    def test_local_match_uses_actual_crop_offset_and_width_based_scales(self):
        self.image[550:580,1100:1200]=(10,20,30)
        templates=LocalTemplates((self.local_hit,))
        obs,_=self.analyze((self.span,),templates)
        self.assertEqual(obs.scene,'movement_preview')
        self.assertEqual(len(obs.actions),1)
        self.assertEqual(obs.actions[0].bbox,(1100,550,100,30))
        self.assertEqual(obs.actions[0].metadata['source_frame_id'],'test-frame')
        self.assertEqual(obs.actions[0].metadata['operation'],'event_advance')
        pixels,options=templates.local_calls[0]
        np.testing.assert_array_equal(pixels,self.image[532:598,1082:1218])
        self.assertEqual(options['threshold'],.88)
        self.assertEqual(options['maximum'],2)
        self.assertEqual(len(options['scales']),17)
        self.assertAlmostEqual(min(options['scales']),.92)
        self.assertAlmostEqual(max(options['scales']),1.08)

    def test_duplicate_local_template_matches_refuse_entry(self):
        templates=LocalTemplates((self.local_hit,replace(self.local_hit,bbox=(19,18,100,30))))
        obs,_=self.analyze((self.span,),templates)
        self.assertEqual(obs.scene,'movement_preview')
        self.assertFalse(obs.actions)

    def test_invalid_local_scores_scales_and_boxes_do_not_ground_action(self):
        invalid=[replace(self.local_hit,confidence=v) for v in (.879,float('nan'),float('inf'),1.01)]
        invalid += [replace(self.local_hit,scale=v) for v in (0,-1,float('nan'),float('inf'))]
        invalid += [replace(self.local_hit,bbox=v) for v in ((-1,18,100,30),(18,18,0,30),
            (18,18,100,-1),(40,18,100,30),(18,45,100,30),(float('nan'),18,100,30))]
        for hit in invalid:
            with self.subTest(hit=hit):
                obs,_=self.analyze((self.span,),LocalTemplates((hit,)))
                self.assertEqual(obs.scene,'movement_preview')
                self.assertFalse(obs.actions)

    def test_match_far_from_ocr_center_is_rejected(self):
        hit=replace(self.local_hit,bbox=(0,0,50,20))
        obs,_=self.analyze((self.span,),LocalTemplates((hit,)))
        self.assertFalse(obs.actions)

    def test_bad_or_unrelated_ocr_is_not_promoted_to_panel_hint(self):
        for span in (replace(self.span,text='出发前'),replace(self.span,text='前往'),
                     replace(self.span,confidence=.79),replace(self.span,confidence=float('nan')),
                     replace(self.span,bbox=(10,550,100,30)),replace(self.span,bbox=(1100,100,100,30)),
                     replace(self.span,bbox=(1200,550,100,30)),replace(self.span,bbox=(1100,550,0,30)),
                     replace(self.span,bbox=(1100,float('nan'),100,30))):
            with self.subTest(span=span):
                templates=LocalTemplates((self.local_hit,))
                obs,_=self.analyze((span,),templates)
                self.assertEqual(obs.scene,'unknown')
                self.assertFalse(obs.actions)
                self.assertFalse(templates.local_calls)

    def test_known_battle_and_result_scenes_still_win(self):
        cases=(('Roguelike@StartAction.png','battle_start'),('BattleOfficiallyBegin.png','battle'),
               ('BlackFlow@Roguelike@GamePass.png','ending'),('BlackFlow@Roguelike@GetDrop.png','reward'))
        for name,scene in cases:
            with self.subTest(scene=scene):
                templates=LocalTemplates((self.local_hit,),{name:[TemplateHit(name,.99,(900,300,80,30),1)]})
                obs,_=self.analyze((self.span,),templates)
                self.assertEqual(obs.scene,scene)
                self.assertFalse(any(a.action_id=='map:enter_preview' for a in obs.actions))
        obs,_=self.analyze((self.span,OCRSpan('开始行动',.99,(950,620,130,35))),LocalTemplates((self.local_hit,)))
        self.assertEqual(obs.scene,'battle_start')
        self.assertFalse(obs.actions)


if __name__=='__main__':
    unittest.main()
