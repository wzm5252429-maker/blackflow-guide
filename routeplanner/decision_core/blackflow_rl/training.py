from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np

from .agents import HeuristicEvaluator, choose_heuristic_action, choose_random_action
from .domain import GameState
from .features import EncodedState, FeatureEncoder, stack_encoded
from .mcts import MCTSConfig, PUCTMCTS
from .simulator import BlackflowSimulator


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "训练需要 PyTorch。请先执行 `py -3.13 -m pip install -r requirements.txt`。"
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 20260831
    episodes: int = 5
    simulations_per_move: int = 8
    replay_capacity: int = 20_000
    batch_size: int = 32
    updates_per_episode: int = 2
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 5.0
    policy_temperature: float = 1.0
    hidden_dim: int = 64
    device: str = "cpu"
    max_steps_per_episode: int = 300

    def __post_init__(self) -> None:
        positive_ints = (
            "episodes",
            "simulations_per_move",
            "replay_capacity",
            "batch_size",
            "updates_per_episode",
            "max_steps_per_episode",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")


@dataclass(frozen=True, slots=True)
class ReplaySample:
    encoded: EncodedState
    policy: np.ndarray
    value_target: float


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    seed: int
    steps: int
    total_reward: float
    chase_count: int
    samples: int


@dataclass(frozen=True, slots=True)
class LossStats:
    total: float
    policy: float
    value: float
    entropy: float


class TorchPolicyValueEvaluator:
    """Adapter from the graph network to the dependency-free MCTS protocol."""

    def __init__(self, model: Any, encoder: FeatureEncoder) -> None:
        self.model = model
        self.encoder = encoder

    def evaluate(
        self,
        state: GameState,
        legal_action_ids: tuple[int, ...],
    ) -> tuple[Mapping[int, float], float]:
        torch = _torch()
        encoded = self.encoder.encode(state)
        tensors = encoded.as_torch(str(self.model.device))
        logits, value = self.model.predict(**tensors)
        if not legal_action_ids:
            return {}, float(value.item())
        indices = torch.as_tensor(legal_action_ids, device=logits.device, dtype=torch.long)
        probabilities = torch.softmax(logits.index_select(0, indices), dim=0)
        priors = {
            action_id: float(probability)
            for action_id, probability in zip(
                legal_action_ids, probabilities.detach().cpu().tolist(), strict=True
            )
        }
        return priors, float(value.item())


class Trainer:
    CHECKPOINT_FORMAT = 2

    def __init__(
        self,
        simulator: BlackflowSimulator,
        config: TrainingConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        torch = _torch()
        from .network import GraphPolicyValueNetwork, NetworkConfig

        self.simulator = simulator
        if simulator.simulation_profile != "synthetic":
            raise ValueError(
                "端到端 rollout 目前依赖未验证的地图/事件先验；"
                "训练必须显式构造 synthetic profile，且结果不得称为真实规则训练"
            )
        self.config = config or TrainingConfig()
        self.encoder = FeatureEncoder(simulator, full_observability=False)
        torch.manual_seed(self.config.seed)
        self.rng = random.Random(self.config.seed)
        self.numpy_rng = np.random.default_rng(self.config.seed)
        if model is None:
            model = GraphPolicyValueNetwork(
                NetworkConfig(
                    node_feature_dim=self.encoder.node_feature_dim,
                    global_feature_dim=self.encoder.global_feature_dim,
                    option_feature_dim=self.encoder.option_feature_dim,
                    hidden_dim=self.config.hidden_dim,
                )
            )
        self.model = model.to(self.config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.replay: deque[ReplaySample] = deque(maxlen=self.config.replay_capacity)
        self.episodes_completed = 0

    def make_mcts(
        self,
        seed: int,
        *,
        training: bool,
        simulations: int | None = None,
    ) -> PUCTMCTS[GameState]:
        return PUCTMCTS(
            self.simulator,
            TorchPolicyValueEvaluator(self.model, self.encoder),
            MCTSConfig(
                num_simulations=(
                    self.config.simulations_per_move
                    if simulations is None
                    else simulations
                ),
                c_puct=1.5,
                gamma=self.simulator.ruleset.objective.discount,
                reward_scale=self.simulator.ruleset.objective.value_scale,
                max_depth=160,
                temperature=self.config.policy_temperature if training else 0.0,
                add_root_dirichlet_noise=training,
                dirichlet_alpha=0.3,
                dirichlet_fraction=0.20,
                seed=seed,
            ),
        )

    def collect_episode(self, seed: int) -> EpisodeStats:
        state = self.simulator.reset(seed)
        trajectory: list[tuple[EncodedState, np.ndarray, float]] = []
        mcts = self.make_mcts(seed ^ 0x5F3759DF, training=True)
        for _ in range(self.config.max_steps_per_episode):
            if state.terminal:
                break
            # Search over an expected hidden world, never over the episode's
            # unrevealed true contents.  The chosen action is then applied to
            # the real episode state.
            planning_state = self.simulator.belief_state(state)
            result = mcts.search(planning_state)
            if result.selected_action is None:
                raise RuntimeError("MCTS returned no action for a non-terminal state")
            # The policy target came from this belief state, so pair it with
            # the exact same observable representation.  The encoder is also
            # tested to make real/belief observations equivalent.
            encoded = self.encoder.encode(planning_state)
            transition = self.simulator.transition(state, result.selected_action)
            trajectory.append(
                (
                    encoded,
                    np.asarray(result.visit_policy, dtype=np.float32),
                    transition.reward,
                )
            )
            state = transition.next_state
        else:
            raise RuntimeError("episode exceeded max_steps_per_episode")

        future_return = 0.0
        samples: list[ReplaySample] = []
        objective = self.simulator.ruleset.objective
        for encoded, policy, reward in reversed(trajectory):
            future_return = reward + objective.discount * future_return
            value_target = float(
                np.clip(future_return / objective.value_scale, -1.0, 1.0)
            )
            samples.append(ReplaySample(encoded, policy, value_target))
        self.replay.extend(reversed(samples))
        return EpisodeStats(
            seed=seed,
            steps=state.step_count,
            total_reward=state.total_reward,
            chase_count=state.chase_count,
            samples=len(samples),
        )

    def train_batch(self) -> LossStats | None:
        if not self.replay:
            return None
        torch = _torch()
        sample_count = min(self.config.batch_size, len(self.replay))
        batch = self.rng.sample(list(self.replay), sample_count)
        arrays = stack_encoded([sample.encoded for sample in batch])
        tensors = {
            name: torch.as_tensor(value, device=self.config.device)
            for name, value in arrays.items()
        }
        target_policy = torch.as_tensor(
            np.stack([sample.policy for sample in batch]),
            device=self.config.device,
        )
        target_value = torch.as_tensor(
            [sample.value_target for sample in batch],
            device=self.config.device,
            dtype=torch.float32,
        )

        self.model.train()
        logits, values = self.model(**tensors)
        log_policy = torch.log_softmax(logits, dim=1)
        policy_loss = -(target_policy * log_policy).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(values, target_value)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        probabilities = torch.softmax(logits, dim=1)
        entropy = -(probabilities * log_policy).sum(dim=1).mean()
        return LossStats(
            total=float(loss.detach().cpu()),
            policy=float(policy_loss.detach().cpu()),
            value=float(value_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
        )

    def train(self, episodes: int | None = None) -> list[dict[str, Any]]:
        episode_count = episodes if episodes is not None else self.config.episodes
        reports: list[dict[str, Any]] = []
        for _ in range(episode_count):
            episode_seed = self.rng.getrandbits(63)
            episode_stats = self.collect_episode(episode_seed)
            losses = [
                loss
                for _ in range(self.config.updates_per_episode)
                if (loss := self.train_batch()) is not None
            ]
            self.episodes_completed += 1
            report: dict[str, Any] = asdict(episode_stats)
            report["episode"] = self.episodes_completed
            report["replay_size"] = len(self.replay)
            if losses:
                report["loss"] = sum(loss.total for loss in losses) / len(losses)
                report["policy_loss"] = sum(loss.policy for loss in losses) / len(losses)
                report["value_loss"] = sum(loss.value for loss in losses) / len(losses)
            reports.append(report)
        return reports

    def save_checkpoint(self, path: str | Path) -> Path:
        torch = _torch()
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        payload = {
            "format_version": self.CHECKPOINT_FORMAT,
            "ruleset_id": self.simulator.ruleset.ruleset_id,
            "rules_sha256": self.simulator.ruleset.sha256,
            "simulation_profile": self.simulator.simulation_profile,
            "environment_sha256": self.simulator.environment_sha256,
            "training_config": asdict(self.config),
            "network_config": asdict(self.model.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episodes_completed": self.episodes_completed,
            "python_random_state": self.rng.getstate(),
            "numpy_random_state": self.numpy_rng.bit_generator.state,
            "torch_random_state": torch.random.get_rng_state(),
            "replay": list(self.replay),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        simulator: BlackflowSimulator,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> "Trainer":
        torch = _torch()
        from .network import GraphPolicyValueNetwork, NetworkConfig

        source = Path(path).resolve()
        try:
            payload = torch.load(source, map_location=device or "cpu", weights_only=False)
        except TypeError:  # older torch
            payload = torch.load(source, map_location=device or "cpu")
        if payload.get("format_version") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported trainer checkpoint format")
        if payload.get("rules_sha256") != simulator.ruleset.sha256:
            raise ValueError("checkpoint rules SHA does not match current rules")
        if payload.get("simulation_profile") != simulator.simulation_profile:
            raise ValueError("checkpoint simulation profile does not match current environment")
        if payload.get("environment_sha256") != simulator.environment_sha256:
            raise ValueError("checkpoint environment SHA does not match current semantics")
        config_data = dict(payload["training_config"])
        if device is not None:
            config_data["device"] = device
        config = TrainingConfig(**config_data)
        model = GraphPolicyValueNetwork(NetworkConfig(**payload["network_config"]))
        model.load_state_dict(payload["model_state_dict"])
        trainer = cls(simulator, config, model=model)
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        trainer.episodes_completed = int(payload["episodes_completed"])
        trainer.rng.setstate(payload["python_random_state"])
        trainer.numpy_rng.bit_generator.state = payload["numpy_random_state"]
        torch.random.set_rng_state(payload["torch_random_state"])
        trainer.replay.extend(payload.get("replay", []))
        return trainer


def run_episode(
    simulator: BlackflowSimulator,
    seed: int,
    *,
    policy: str,
    trainer: Trainer | None = None,
    simulations: int = 32,
) -> EpisodeStats:
    state = simulator.reset(seed)
    rng = random.Random(seed ^ 0xA5A5A5A5)
    heuristic = HeuristicEvaluator(simulator)
    if policy == "mcts":
        if trainer is None:
            mcts = PUCTMCTS(
                simulator,
                heuristic,
                MCTSConfig(
                    num_simulations=simulations,
                    gamma=simulator.ruleset.objective.discount,
                    reward_scale=simulator.ruleset.objective.value_scale,
                    temperature=0.0,
                    seed=seed,
                ),
            )
        else:
            mcts = trainer.make_mcts(
                seed, training=False, simulations=simulations
            )
    else:
        mcts = None

    while not state.terminal:
        if policy == "random":
            action_id = choose_random_action(simulator, state, rng)
        elif policy == "heuristic":
            action_id = choose_heuristic_action(heuristic, state)
        elif policy == "mcts":
            assert mcts is not None
            result = mcts.search(simulator.belief_state(state), temperature=0.0, add_root_noise=False)
            if result.selected_action is None:
                raise RuntimeError("MCTS returned no action")
            action_id = result.selected_action
        else:
            raise ValueError(f"unknown policy: {policy}")
        state = simulator.transition(state, action_id).next_state

    return EpisodeStats(
        seed=seed,
        steps=state.step_count,
        total_reward=state.total_reward,
        chase_count=state.chase_count,
        samples=0,
    )


def evaluate_policies(
    simulator: BlackflowSimulator,
    seeds: Iterable[int],
    *,
    policies: tuple[str, ...] = ("random", "heuristic", "mcts"),
    trainer: Trainer | None = None,
    simulations: int = 32,
) -> dict[str, dict[str, float]]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("evaluation needs at least one seed")
    result: dict[str, dict[str, float]] = {}
    for policy in policies:
        episodes = [
            run_episode(
                simulator,
                seed,
                policy=policy,
                trainer=trainer,
                simulations=simulations,
            )
            for seed in seed_list
        ]
        result[policy] = {
            "episodes": float(len(episodes)),
            "mean_reward": float(np.mean([item.total_reward for item in episodes])),
            "mean_steps": float(np.mean([item.steps for item in episodes])),
            "mean_chases": float(np.mean([item.chase_count for item in episodes])),
        }
    return result
