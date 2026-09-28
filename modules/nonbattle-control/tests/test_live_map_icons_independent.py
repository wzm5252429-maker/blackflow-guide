"""Independent map-icon graph boundaries; injected hits, no desktop or OCR."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from blackflow_live.vision import OCRSpan, TemplateHit, VisionPipeline
from tests.test_live_vision import Templates, observer


class MapTemplates(Templates):
    def __init__(self,specs,hits,labels=None):
        super().__init__(hits)
        self.node_specs=specs
        self.labels=labels or {}
        self.requests=[]
    def match(self,image,name,**options):
        self.requests.append((name,options))
        # Deliberately return unfiltered hits to test the graph's own boundary.
        return self.hits.get(name,[])


class IndependentMapIconTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((720,1280,3),np.uint8)

    def pipeline(self,extra=(),*,labels=None,role='ordinary',threshold=.8):
        specs=[{'file':'a','role':'ordinary','node_type':'battle_normal','threshold':.8},
               {'file':'b','role':'ordinary','node_type':'rest','threshold':.8},
               {'file':'extra','role':role,'node_type':'shop','threshold':threshold}]
        templates=MapTemplates(specs,{'a':[TemplateHit('a',.96,(180,280,40,40),1)],
            'b':[TemplateHit('b',.95,(280,280,40,40),1)],'extra':list(extra)},labels)
        pipeline=object.__new__(VisionPipeline)
        pipeline.templates=templates
        pipeline.ocr=SimpleNamespace()
        pipeline.corridor=SimpleNamespace(config={'decision':{'probability_threshold':.89}},
                                          score=lambda image,pairs:[.99]*len(pairs))
        return pipeline

    def test_ordinary_threshold_is_not_manifest_point_eight(self):
        p=self.pipeline((TemplateHit('extra',.879,(380,280,40,40),1),))
        nodes,_,_,_=p._map(self.image,())
        self.assertEqual(len(nodes),2)
        self.assertTrue(all(options['threshold']==.88 for _,options in p.templates.requests))

    def test_special_and_empty_thresholds_remain_independent(self):
        for role,configured,score,accepted in (('special',.8,.79,False),('special',.92,.90,False),
                                              ('empty',.5,.77,False),('empty',.5,.79,True)):
            with self.subTest(role=role,score=score):
                p=self.pipeline((TemplateHit('extra',score,(380,280,40,40),1),),role=role,threshold=configured)
                nodes,_,_,_=p._map(self.image,())
                self.assertEqual(len(nodes),3 if accepted else 2)

    def test_conflicting_icon_types_reject_whole_graph_despite_score_order(self):
        for score in (.88,.999):
            p=self.pipeline((TemplateHit('extra',score,(181,280,40,40),1),))
            nodes,edges,current,notes=p._map(self.image,())
            self.assertEqual((nodes,edges,current),((),(),None))
            self.assertIn('conflicting_map_node_types',notes)

    def test_conflicting_complete_ocr_label_rejects_icon_graph(self):
        p=self.pipeline(labels={'安全的角落':'rest'})
        # Text center (200,329) maps to node center (200,300) at scale 1.
        result=p._map(self.image,(OCRSpan('安全的角落',.99,(170,319,60,20)),))
        self.assertIn('conflicting_map_node_types',result[3])
        self.assertFalse(result[0])

    def test_shop_and_boss_aliases_do_not_create_false_conflict(self):
        for kind,label_kind,expected in (('shop','scrap_shop','SCRAP_SHOP'),
                ('battle_boss','battle_mid_boss_shwksc','BATTLE_BOSS')):
            with self.subTest(kind=kind):
                p=self.pipeline(labels={'同类节点':label_kind})
                p.templates.node_specs[0]['node_type']=kind
                nodes,_,_,notes=p._map(self.image,(OCRSpan('同类节点',.99,(170,319,60,20)),))
                self.assertNotIn('conflicting_map_node_types',notes)
                self.assertEqual(len(nodes),2)
                self.assertIn(expected,{n.node_type for n in nodes})

    def test_hidden_types_stay_unrevealed(self):
        for kind in ('hide_battle','hide_invisible','unclassified'):
            with self.subTest(kind=kind):
                p=self.pipeline()
                p.templates.node_specs[0]['node_type']=kind
                nodes,_,_,_=p._map(self.image,())
                hidden=next(n for n in nodes if n.node_type==kind)
                self.assertFalse(hidden.revealed)
                self.assertFalse(hidden.completed)

    def test_nonfinite_out_of_bounds_and_nonpositive_hits_never_enter_graph(self):
        normal=TemplateHit('extra',.99,(380,280,40,40),1)
        invalid=[replace(normal,confidence=score) for score in (float('nan'),float('inf'),-1,1.01)]
        invalid += [replace(normal,bbox=box) for box in ((-1,280,40,40),(1270,280,40,40),
            (380,700,40,40),(380,280,0,40),(380,280,40,-1),(float('nan'),280,40,40))]
        invalid += [replace(normal,scale=scale) for scale in (0,-1,float('nan'),float('inf'))]
        for hit in invalid:
            with self.subTest(hit=hit):
                nodes,_,_,_=self.pipeline((hit,))._map(self.image,())
                self.assertEqual(len(nodes),2)

    def test_map_icons_do_not_override_battle_or_overlay_scene(self):
        markers=('Roguelike@StartAction.png','BattleOfficiallyBegin.png',
                 'BlackFlow@Roguelike@CloseEvent.png','BlackFlow@Roguelike@GetDrop.png',
                 'BlackFlow@Roguelike@CultivateConfirm.png','BlackFlow@Roguelike@MovementInventoryCollapse.png')
        for name in markers:
            with self.subTest(marker=name):
                templates=Templates({name:[TemplateHit(name,.99,(900,300,40,40),1)],
                    'BlackFlow@Roguelike@MapZoomIn.png':[TemplateHit('map',.99,(10,500,30,30),1)]})
                p=observer(templates)
                p._map=Mock(side_effect=AssertionError('Overlay must not enter map recognizer'))
                result=p.analyze(self.image,(),frame_id='one',captured_at=1.)
                self.assertNotEqual(result.scene,'map')
                p._map.assert_not_called()


if __name__=='__main__':
    unittest.main()
