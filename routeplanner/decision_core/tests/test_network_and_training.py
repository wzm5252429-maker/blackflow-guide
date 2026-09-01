from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from blackflow_rl.features import FeatureEncoder
from blackflow_rl.mapgen import MapGenerator, MapGeneratorConfig
from blackflow_rl.network import GraphPolicyValueNetwork, NetworkConfig
from blackflow_rl.simulator import BlackflowSimulator
from blackflow_rl.training import ReplaySample, Trainer, TrainingConfig


class NetworkAndTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = BlackflowSimulator(
            map_generator=MapGenerator(
                config=MapGeneratorConfig(
                    allow_synthetic_map_sampling=True,
                    allow_synthetic_event_effects=True,
                )
            )
        )
        self.encoder = FeatureEncoder(self.env)

    def test_encoder_and_network_mask_shapes(self) -> None:
        state = self.env.reset(123)
        encoded = self.encoder.encode(state)
        belief_encoded = self.encoder.encode(self.env.belief_state(state))
        for field_name in encoded.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(encoded, field_name), getattr(belief_encoded, field_name)
            )
        model = GraphPolicyValueNetwork(
            NetworkConfig(
                self.encoder.node_feature_dim,
                self.encoder.global_feature_dim,
                self.encoder.option_feature_dim,
                hidden_dim=32,
                num_message_passing_layers=2,
            )
        )
        logits, value = model.predict(**encoded.as_torch())
        self.assertEqual(tuple(logits.shape), (self.env.action_size,))
        self.assertEqual(value.ndim, 0)
        legal = set(self.env.legal_action_ids(state))
        for action_id in range(self.env.action_size):
            if action_id not in legal:
                self.assertLess(float(logits[action_id]), -1e20)
        self.assertTrue(torch.isfinite(value))

    def test_one_training_batch_and_checkpoint_roundtrip(self) -> None:
        config = TrainingConfig(
            episodes=1,
            simulations_per_move=1,
            replay_capacity=8,
            batch_size=1,
            updates_per_episode=1,
            hidden_dim=32,
            seed=11,
        )
        trainer = Trainer(self.env, config)
        state = self.env.reset(11)
        encoded = trainer.encoder.encode(state)
        policy = np.zeros(self.env.action_size, dtype=np.float32)
        policy[self.env.legal_action_ids(state)[0]] = 1.0
        trainer.replay.append(ReplaySample(encoded, policy, 0.25))
        loss = trainer.train_batch()
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss.total))

        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save_checkpoint(Path(directory) / "model.pt")
            restored = Trainer.load_checkpoint(self.env, path)
            self.assertEqual(len(restored.replay), 1)
            self.assertEqual(
                restored.make_mcts(
                    1, training=False, simulations=7
                ).config.num_simulations,
                7,
            )
            first = trainer.model.predict(**encoded.as_torch())[0]
            second = restored.model.predict(**encoded.as_torch())[0]
            self.assertTrue(torch.allclose(first, second))

            changed_env = BlackflowSimulator(
                map_generator=MapGenerator(
                    config=MapGeneratorConfig(
                        allow_synthetic_map_sampling=True,
                        allow_synthetic_event_effects=True,
                        door_pair_probability=0.75,
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "environment SHA"):
                Trainer.load_checkpoint(changed_env, path)


if __name__ == "__main__":
    unittest.main()
