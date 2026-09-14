"""Real screenshot-policy boundary tests; no game client input is issued."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blackflow_live.models import LiveObservation, ObservedAction, ObservedNode
from blackflow_live.policy import CurrentNeuralPolicy, ROOT, action_is_safe


def button(action_id="continue", label="继续", operation="advance", **metadata):
    return ObservedAction(action_id, label, "ui", (500., 400., 80., 32.), .98,
                          metadata={"operation":operation, **metadata})


def observation(scene="event", actions=None, **kwargs):
    return LiveObservation("test-frame", 1., scene, .99,
                           actions=tuple(actions or (button(),)),
                           metadata={"image_width":1280,"image_height":720}, **kwargs)


def route_observation():
    nodes = (ObservedNode("a","START",0,0,(50,300,50,50)),
             ObservedNode("b","WISH",0,1,(200,300,50,50)),
             ObservedNode("c","SCRAP_SHOP",1,0,(50,450,50,50)))
    actions = tuple(ObservedAction("node:"+n.node_id,n.node_type,"map_node",n.bbox,
        target_node_id=n.node_id,metadata={"operation":"move","movement_cost":1}) for n in nodes[1:])
    return observation("map",actions,nodes=nodes,edges=(("a","b"),("a","c")),current_node_id="a",floor=1,
                       resources={"hp":8,"max_hp":8,"gold":12,"hope":3,"parts":2,"relics":1,"action_points":5})


class LiveScopeTests(unittest.TestCase):
    def test_start_battle_label_cannot_hide_behind_ui_kind(self):
        action = button(label="开始战斗")
        self.assertFalse(action_is_safe(action, observation(actions=(action,))))

    def test_other_ending_semantics_and_source_expedition_are_masked(self):
        for metadata in ({"item_id":"rogue_6_relic_final_2"},
                         {"ending_id":2},{"add_items":["alpha"]},{"operation":"expedition_source"}):
            action = button(**metadata)
            self.assertFalse(action_is_safe(action, observation(actions=(action,))))

    def test_ordinary_lake_offering_is_not_an_ending_restriction(self):
        action=button(operation="event",choice_id="choice_ro6_bat5_7")
        self.assertTrue(action_is_safe(action,observation(actions=(action,))))

    def test_unaffordable_or_unobserved_price_purchase_is_masked(self):
        for metadata in ({},{"price":13},{"price":-1}):
            action = button(operation="purchase",**metadata)
            self.assertFalse(action_is_safe(action,observation(actions=(action,),resources={"gold":12})))
        action = button(operation="purchase",price=10)
        self.assertTrue(action_is_safe(action,observation(actions=(action,),resources={"gold":12})))

    def test_outside_frame_nonfinite_low_confidence_and_disabled_are_masked(self):
        for action in (replace(button(),bbox=(1270,0,30,30)),replace(button(),confidence=float("nan")),
                       replace(button(),confidence=.5),replace(button(),enabled=False)):
            self.assertFalse(action_is_safe(action,observation(actions=(action,))))

    def test_preview_does_not_spend_or_require_the_displayed_cost(self):
        action=button(operation="event",preview_only=True,hope_cost=5)
        self.assertTrue(action_is_safe(action,observation(actions=(action,))))
        self.assertFalse(action_is_safe(replace(action,metadata={"operation":"recruit_reserve","hope_cost":5}),observation()))
        self.assertTrue(action_is_safe(replace(action,metadata={"operation":"recruit_temporary","hope_cost":0}),observation()))

    def test_battle_and_confirmed_first_ending_stop_before_loading_weights(self):
        policy = CurrentNeuralPolicy()
        with patch.object(policy,"load",side_effect=AssertionError("must not load")):
            self.assertIsNone(policy.select(observation("battle")).action)
            self.assertIsNone(policy.select(observation(ending_first_confirmed=True)).action)

    def test_missing_selection_never_substitutes_random_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = CurrentNeuralPolicy(selection_path=Path(directory)/"missing.json")
            decision = policy.select(observation())
        self.assertIsNone(decision.action)
        self.assertFalse(decision.neural)
        self.assertIn("neural_weights_unavailable",decision.reason)


class CurrentWeightInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selection = json.loads((ROOT/"data/policies/current_neural_controller.json").read_text(encoding="utf-8"))
        if not all((ROOT/selection[key]).is_file() for key in ("checkpoint","menu_checkpoint","profile")):
            raise unittest.SkipTest("Selected local neural artifacts are not distributed with this checkout")
        cls.policy = CurrentNeuralPolicy()
        with patch("blackflow_rl.simulator.BlackflowSimulator.reset",side_effect=AssertionError("no simulated reset")), \
             patch("blackflow_rl.simulator.BlackflowSimulator.transition",side_effect=AssertionError("no simulated transition")):
            if not cls.policy.load():
                raise AssertionError(cls.policy.load_error)

    def test_current_route_and_menu_forward_use_observed_actions(self):
        for obs in (route_observation(),observation(actions=(button(),button("leave","离开","leave")))):
            with patch("blackflow_rl.simulator.BlackflowSimulator.transition",side_effect=AssertionError("no rollout")):
                decision = self.policy.select(obs)
            self.assertTrue(decision.neural,decision.reason)
            self.assertIn(decision.action,obs.actions)
            self.assertGreater(decision.confidence,0)

    def test_map_zoom_uses_neural_menu_without_fabricating_route_state(self):
        obs = observation("map",actions=(button("zoom","缩小地图","event_advance"),))
        decision = self.policy.select(obs)
        self.assertTrue(decision.neural,decision.reason)
        self.assertEqual(decision.action_id,"zoom")
        self.assertEqual(obs.resources,{})
        self.assertIsNone(obs.current_node_id)

    def test_route_refuses_missing_real_resources_and_current_node(self):
        obs = route_observation()
        for incomplete in (replace(obs,current_node_id=None),replace(obs,resources={"action_points":5}),replace(obs,floor=6)):
            decision = self.policy.select(incomplete)
            self.assertIsNone(decision.action)
            self.assertFalse(decision.neural)
            self.assertIn("route_",decision.reason)

    def test_encoder_preserves_actual_resources_and_does_not_advance_them(self):
        obs = route_observation()
        encoded,indices,_ = self.policy.encoder.encode(obs,obs.actions)
        self.assertAlmostEqual(float(encoded.global_features[6]),12/50,places=6)
        self.assertEqual(float(encoded.global_features[2]),1)
        self.assertIn("team_strength",self.policy.encoder.last_missing_fields)
        self.assertTrue(encoded.action_mask[indices[obs.actions[0].action_id]])
        self.assertEqual(obs.resources["gold"],12)

    def test_battle_is_masked_even_when_menu_network_prefers_it(self):
        obs = observation(actions=(button("fight","开始战斗"),button()))
        decision = self.policy.select(obs)
        self.assertTrue(decision.neural,decision.reason)
        self.assertEqual(decision.action_id,"continue")

    def test_menu_selection_is_the_actual_frozen_network_argmax(self):
        import torch
        obs = observation(actions=(button(),button("leave","离开","leave")))
        encoded,indices,_ = self.policy.encoder.encode(obs,obs.actions)
        tensors={name:value.unsqueeze(0) for name,value in encoded.as_torch().items()}
        with torch.inference_mode():
            logits,_=self.policy.menu_model(**tensors)
        best=max(obs.actions,key=lambda action:float(logits[0,indices[action.action_id]]))
        self.assertEqual(self.policy.select(obs).action_id,best.action_id)

    def test_duplicate_action_identity_and_unknown_operations_are_refused(self):
        duplicate=observation(actions=(button(),button()))
        self.assertIsNone(self.policy.select(duplicate).action)
        unknown=observation(actions=(button(operation="unclassified"),))
        self.assertIsNone(self.policy.select(unknown).action)

    def test_exact_observed_shop_name_encodes_identity_and_actual_price(self):
        layout=self.policy.encoder.layout
        item=layout.simulator.economy.catalog.items["rogue_6_scrap_G_04"]
        action=button("buy",item.name,"purchase",item_name=item.name,price=7,quantity=1)
        obs=observation("shop",(action,),resources={"gold":8})
        encoded,indices,_=self.policy.encoder.encode(obs,obs.actions)
        row=encoded.option_features[0]
        identity_start=layout.option_feature_dim-len(layout.item_identities)
        self.assertEqual(float(row[identity_start+layout._item_identity_index[item.canonical_id]]),1)
        self.assertAlmostEqual(float(row[identity_start-12]),7/30,places=6)
        decision=self.policy.select(obs)
        self.assertTrue(decision.neural,decision.reason)
        self.assertIs(decision.action,action)
        self.assertNotIn("item_id",action.metadata)

    def test_conflicting_item_name_and_id_are_never_clicked(self):
        layout=self.policy.encoder.layout
        item=layout.simulator.economy.catalog.items["rogue_6_scrap_G_04"]
        action=button(operation="purchase",item_name=item.name,item_id="rogue_6_recruit_ticket_tank",price=1)
        self.assertIsNone(self.policy.select(observation("shop",(action,),resources={"gold":5})).action)

    def test_observed_mechanist_and_hope_cost_reach_checkpoint_features(self):
        from blackflow_rl.features import RESOURCE_FIELDS,RESOURCE_SCALES
        layout=self.policy.encoder.layout
        adapter=self.policy.encoder.metadata_adapter
        operator=adapter.operators["char_4230_mcnist"]["name"]
        action=button(operation="recruit_temporary",operator_name=operator,hope_cost=2)
        obs=observation("recruitment",(action,),resources={"hope":3})
        encoded,_,_=self.policy.encoder.encode(obs,obs.actions)
        physical=layout.option_feature_dim-len(layout.item_identities)-12
        self.assertEqual(float(encoded.option_features[0,physical+9]),1)
        hope=RESOURCE_FIELDS.index("hope")
        self.assertAlmostEqual(float(encoded.option_features[0,hope]),-2/RESOURCE_SCALES[hope],places=6)
        self.assertIsNotNone(self.policy.select(obs).action)
        self.assertIsNone(self.policy.select(replace(obs,resources={"hope":1})).action)
        self.assertEqual(obs.resources["hope"],3)

    def test_recruit_ticket_and_upgrade_ticket_keep_trained_distinctions(self):
        layout=self.policy.encoder.layout
        actions=(button("ticket",operation="select_recruit_ticket",item_id="rogue_6_recruit_ticket_tank"),
                 button("upgrade",operation="take",item_id="rogue_6_upgrade_ticket_all"))
        obs=observation("reward",actions)
        encoded,_,_=self.policy.encoder.encode(obs,obs.actions)
        physical=layout.option_feature_dim-len(layout.item_identities)-12
        self.assertEqual(float(encoded.option_features[0,physical+10]),1)
        self.assertEqual(float(encoded.option_features[1,physical+11]),1)

    def test_complete_current_shelf_encodes_unaffordable_visible_stock(self):
        from blackflow_rl.features import OBSERVED_NODE_LABELS,SHOP_SCALAR_FIELDS,SHOP_CATEGORIES
        node=ObservedNode("shop","SCRAP_SHOP",0,0,(100,200,50,50))
        action=button(operation="purchase",item_id="rogue_6_scrap_G_04",price=15,slot_id="slot-1")
        obs=observation("shop",(action,),nodes=(node,),current_node_id="shop",floor=1,resources={"gold":1})
        obs=replace(obs,metadata={**obs.metadata,"shop_state":{"node_id":"shop","shelf_complete":True,"source_frame_id":obs.frame_id}})
        encoded,_,_=self.policy.encoder.encode(obs,())
        layout=self.policy.encoder.layout
        offset=len(OBSERVED_NODE_LABELS)+layout.NODE_SCALAR_DIM
        self.assertEqual(float(encoded.node_features[0,offset]),1)
        self.assertAlmostEqual(float(encoded.node_features[0,offset+len(SHOP_SCALAR_FIELDS)+SHOP_CATEGORIES.index("GOODS")]),.1)
        self.assertTrue(encoded.option_observation_mask[0])
        self.assertFalse(encoded.action_mask.any())
        stale=replace(obs,metadata={**obs.metadata,"shop_state":{**obs.metadata["shop_state"],"source_frame_id":"old"}})
        stale_encoded,_,_=self.policy.encoder.encode(stale,())
        self.assertFalse(stale_encoded.node_features[0,offset:].any())

    def test_readiness_reports_partial_observation_and_no_resource_memory(self):
        self.policy.select(observation())
        report=self.policy.readiness()
        self.assertFalse(report["simulation_used"])
        self.assertFalse(report["last_observation_coverage"]["cross_frame_resource_reuse"])
        self.assertIn("operator",report["limitations"][1])


if __name__ == "__main__":
    unittest.main()
