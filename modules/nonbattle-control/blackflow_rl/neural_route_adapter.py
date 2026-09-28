"""Thirteen learned public action interactions on a frozen direct neural policy.

The final seven inputs describe observable exit timing and shop liquidity. They
do not encode priorities: the head starts at zero, its signs and magnitudes are
learned from concrete actions, and no legal mask or simulator reward is changed.
This adapter is restricted to the five-floor first-ending training profile.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import random

import torch
from torch import nn

from . import neural_concept_adapter as concept
from .features import OBSERVED_NODE_LABELS, INVENTORY_FLAGS
from .ppo import PPOTrainer


FEATURE_NAMES = concept.FEATURE_NAMES + (
    'visible_legal_main_exit_before_floor_five',
    'visible_legal_main_exit_before_floor_five_times_ap_ratio',
    'visible_legal_main_exit_on_floor_five',
    'visible_legal_main_exit_on_floor_five_times_ap_ratio',
    'visible_legal_battle_shop_times_cash_over_fifty',
    'visible_legal_scrap_shop_times_cash_over_fifty',
    'visible_legal_shop_times_natural_appraisal_over_fifty',
)


def implementation_sha256():
    # Include the unchanged shared six-feature implementation in provenance.
    return sha256(Path(__file__).read_bytes() + concept.implementation_sha256().encode()).hexdigest()


def feature_layout(encoder):
    config = encoder.simulator.map_generator.config
    floors = tuple(f for f in sorted(encoder.simulator.ruleset.floors)
                   if f != 6 or config.enable_third_ending)
    if floors != (1, 2, 3, 4, 5):
        raise ValueError('route adapter requires the five-floor first-ending profile')
    result = concept.feature_layout(encoder)
    economy_start = encoder.GLOBAL_BASE_DIM + len(INVENTORY_FLAGS) + encoder.inventory_detail_dim
    result.update(names=FEATURE_NAMES, floor=0, cash=6,
                  natural_appraisal=economy_start+12, portal=economy_start+16,
                  current=len(OBSERVED_NODE_LABELS), exit=len(OBSERVED_NODE_LABELS)+3,
                  battle_shop=OBSERVED_NODE_LABELS.index('BATTLE_SHOP'),
                  scrap_shop=OBSERVED_NODE_LABELS.index('SCRAP_SHOP'),
                  main_floors=floors)
    return result


def action_features(tensors, layout):
    """Return B x actions x 13 using only already public encoded tensors.

Random transport activates at the current-node placeholder, so route destination
features exclude the current node. Portal exits are not main-floor exits. Both
checks affect factual features only, never the simulator's legal-action mask.
    """
    first = concept.action_features(tensors, layout)
    node, glob = tensors['node_features'], tensors['global_features']
    mask = tensors['action_mask'].bool()
    n = node.shape[1]
    result = node.new_zeros((*mask.shape, len(FEATURE_NAMES)))
    result[:, :, :len(concept.FEATURE_NAMES)] = first
    destination = (node[:, :, layout['observed']] * tensors['node_mask'].to(node.dtype)
                   * mask[:, :n] * (1-node[:, :, layout['current']]))
    main_exit = destination * node[:, :, layout['exit']] * (1-glob[:, layout['portal']]).unsqueeze(1)
    final = (glob[:, layout['floor']] >= 1-1e-6).to(node.dtype).unsqueeze(1)
    ap = glob[:, layout['ap']].unsqueeze(1)
    result[:, :n, 6] = main_exit * (1-final)
    result[:, :n, 7] = main_exit * (1-final) * ap
    result[:, :n, 8] = main_exit * final
    result[:, :n, 9] = main_exit * final * ap
    green = destination * node[:, :, layout['battle_shop']]
    yellow = destination * node[:, :, layout['scrap_shop']]
    cash = glob[:, layout['cash']].unsqueeze(1)
    result[:, :n, 10] = green * cash
    result[:, :n, 11] = yellow * cash
    result[:, :n, 12] = (green+yellow) * glob[:, layout['natural_appraisal']].unsqueeze(1)
    return torch.where(mask.unsqueeze(-1), result, torch.zeros_like(result))


class RouteResidualNetwork(nn.Module):
    def __init__(self, base, layout):
        super().__init__()
        self.base, self.layout, self.config = base, layout, base.config
        self.base.requires_grad_(False)
        self.head = nn.Linear(len(FEATURE_NAMES), 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, **tensors):
        with torch.no_grad():
            logits, values = self.base(**tensors)
        adjusted = logits + self.head(action_features(tensors, self.layout)).squeeze(-1)
        adjusted = adjusted.masked_fill(~tensors['action_mask'].bool(), torch.finfo(logits.dtype).min)
        return adjusted, values


class RouteAdapterController(concept.ConceptAdapterController):
    ALGORITHM = 'PUBLIC_ROUTE_RESIDUAL_DIRECT_NEURAL_V2'

    def __init__(self, base_trainer, base_checkpoint):
        self.base = base_trainer
        self.base_checkpoint = Path(base_checkpoint).resolve()
        self.base_checkpoint_sha256 = sha256(self.base_checkpoint.read_bytes()).hexdigest()
        self.simulator, self.encoder, self.config = self.base.simulator, self.base.encoder, self.base.config
        self.policy_constraints = self.base.policy_constraints
        self.layout = feature_layout(self.encoder)
        self.model = RouteResidualNetwork(self.base.model, self.layout).to(self.config.device)
        self.training_seeds = set(self.base.training_seeds)
        self.learning_history = list(self.base.learning_history)
        self.rng = random.Random(self.config.seed)

    @property
    def implementation_sha256(self):
        return implementation_sha256()

    @classmethod
    def load_checkpoint(cls, simulator, path, *, device='cpu'):
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get('algorithm') != cls.ALGORITHM or payload.get('format_version') != 1:
            raise ValueError('not a supported route adapter checkpoint')
        base_path = Path(payload['base_checkpoint'])
        if sha256(base_path.read_bytes()).hexdigest() != payload['base_checkpoint_sha256']:
            raise ValueError('base checkpoint bytes differ')
        if payload.get('feature_code_sha256') != implementation_sha256():
            raise ValueError('adapter feature implementation differs')
        result = cls(PPOTrainer.load_checkpoint(simulator, base_path, device=device), base_path)
        if (payload['environment_sha256'] != simulator.environment_sha256
                or payload['feature_schema_sha256'] != result.encoder.schema_sha256
                or payload['feature_layout'] != result.layout):
            raise ValueError('adapter environment or public feature schema differs')
        result.model.head.load_state_dict(payload['head_state_dict'])
        result.training_seeds.update(payload['training_seeds'])
        if any(PPOTrainer._reserved_seed(s) for s in result.training_seeds):
            raise ValueError('adapter training overlaps reserved evaluation seeds')
        result.learning_history = payload['learning_history']
        result.model.eval()
        return result
