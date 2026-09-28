"""Reward choices bind only to titles in their own observed card."""
from pathlib import Path
import unittest
from blackflow_live.models import ObservedAction
from blackflow_live.vision import DEFAULT_MAA_ROOT, OCRSpan, VisionPipeline, _read_image


def claim(x):
    return ObservedAction(str(x), '收下', 'ui', (x,500,60,30), .99,
                          metadata={'operation':'take','grounded':True})


class RewardCardTests(unittest.TestCase):
    def pipeline(self):
        result=VisionPipeline.__new__(VisionPipeline)
        result.item_names={'甲物品':'甲物品','乙物品':'乙物品'}
        return result

    def test_each_claim_gets_only_its_card_title_and_removes_redundant_preview(self):
        spans=[OCRSpan('甲物品',.99,(280,320,100,30)),OCRSpan('乙物品',.99,(680,320,100,30))]
        previews=[ObservedAction(str(x),'名称','item_preview',(x,320,100,30),.99,
                     metadata={'item_name':name,'preview_only':True}) for x,name in ((280,'甲物品'),(680,'乙物品'))]
        result=self.pipeline()._reward_card_actions([claim(300),claim(700),*previews],spans,1280,720)
        self.assertEqual([a.metadata['item_name'] for a in result],['甲物品','乙物品'])
        self.assertTrue(all(a.metadata['operation']=='take' for a in result))
        self.assertTrue(all('quantity' not in a.metadata and 'price' not in a.metadata for a in result))

    def test_neighbor_card_and_weak_or_unrelated_titles_are_not_borrowed(self):
        for span in (OCRSpan('乙物品',.99,(680,320,100,30)), OCRSpan('甲物品',.89,(280,320,100,30)),
                     OCRSpan('甲物品',.99,(280,60,100,30)), OCRSpan('甲物品',.99,(280,530,100,30))):
            with self.subTest(span=span):
                result=self.pipeline()._reward_card_actions([claim(300)],[span],1280,720)
                self.assertNotIn('item_name',result[0].metadata)

    def test_conflicting_same_column_titles_remain_unknown(self):
        spans=[OCRSpan('甲物品',.99,(280,320,100,30)),OCRSpan('乙物品',.99,(280,380,100,30))]
        result=self.pipeline()._reward_card_actions([claim(300)],spans,1280,720)
        self.assertNotIn('item_name',result[0].metadata)

    def test_card_without_a_claim_keeps_its_preview(self):
        preview=ObservedAction('p','甲物品','item_preview',(280,320,100,30),.99,metadata={'item_name':'甲物品'})
        self.assertEqual(self.pipeline()._reward_card_actions([preview],[],1280,720),[preview])

    def test_narrow_layout_title_belongs_only_to_nearest_claim_column(self):
        title=OCRSpan('乙物品',.99,(360,320,100,30))
        result=self.pipeline()._reward_card_actions([claim(300),claim(380)],[title],720,720)
        self.assertNotIn('item_name',result[0].metadata)
        self.assertEqual(result[1].metadata['item_name'],'乙物品')

    def test_title_between_two_claims_is_ambiguous_and_keeps_preview(self):
        title=OCRSpan('乙物品',.99,(320,320,100,30))
        preview=ObservedAction('p','乙物品','item_preview',title.bbox,.99,metadata={'item_name':'乙物品'})
        result=self.pipeline()._reward_card_actions([claim(300),claim(380),preview],[title],720,720)
        self.assertTrue(all('item_name' not in action.metadata for action in result[:2]))
        self.assertIn(preview,result)


@unittest.skipUnless((DEFAULT_MAA_ROOT/'resource/PaddleOCR/rec/inference.onnx').is_file(),
                     'Installed MAA OCR assets required for real-pixel integration test')
class RecordedRewardTests(unittest.TestCase):
    def test_real_video_reward_buttons_are_model_choices_with_same_card_identity(self):
        from blackflow_live.policy import CurrentNeuralPolicy
        image=_read_image(Path(__file__).parent/'fixtures/live_rewards/two_card_rewards.png')
        observation=VisionPipeline().observe(image,frame_id='video-reward')
        self.assertEqual(observation.scene,'reward')
        claims=[a for a in observation.actions if a.metadata.get('operation')=='take']
        self.assertEqual(len(claims),2)
        named=[a for a in claims if a.metadata.get('item_name')=='标准引擎']
        self.assertEqual(len(named),1)
        self.assertGreater(named[0].bbox[0],image.shape[1]/2)
        self.assertFalse(any(a.kind=='item_preview' for a in observation.actions))
        decision=CurrentNeuralPolicy().select(observation)
        self.assertTrue(decision.neural)
        self.assertIn(decision.action,claims)
        self.assertNotIn('inventory_complete',observation.metadata)


if __name__=='__main__': unittest.main()
