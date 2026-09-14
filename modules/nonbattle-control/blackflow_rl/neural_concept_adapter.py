"""Frozen direct-action network plus a learned public concept-action residual.

No teacher runs at inference. Six feature coefficients are learned from concrete
action labels. The feature extractor sees only FeatureEncoder tensors and never
changes the legal-action mask or queries hidden simulator state.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .features import OBSERVED_NODE_LABELS, INVENTORY_FLAGS, RESOURCE_FIELDS, ITEM_CATEGORIES, OPTION_OPERATIONS
from .ppo import PPOTrainer


FEATURE_NAMES = (
    'deterministic_equipped_wish_concept_relics',
    'deterministic_equipped_sacrifice_concept_relics',
    'concept_relics_times_completed_flag',
    'concept_relics_times_public_ap_ratio',
    'concept_relics_times_equipped_uses_over_ten',
    'equip_option_times_visible_concept_target_potential',
)


def implementation_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def feature_layout(encoder):
    if encoder.full_observability or encoder.SCHEMA_VERSION != 6:
        raise ValueError('concept adapter requires public feature schema 6')
    ids = encoder.item_identities
    moves = encoder.move_identities
    catalog = encoder.simulator.economy.catalog
    detail_start = encoder.GLOBAL_BASE_DIM + len(INVENTORY_FLAGS)
    stride = len(encoder.move_use_bins) + 2
    deterministic = [i for i, identity in enumerate(moves) if not catalog[identity].random_move]
    return {
        'names': FEATURE_NAMES,
        'global_dim': encoder.global_feature_dim,
        'node_dim': encoder.node_feature_dim,
        'option_dim': encoder.option_feature_dim,
        'identity_count': len(ids),
        'p05': encoder.global_feature_dim-len(ids)+ids.index('rogue_6_scrap_P_05'),
        'p06': encoder.global_feature_dim-len(ids)+ids.index('rogue_6_scrap_P_06'),
        'wish': OBSERVED_NODE_LABELS.index('WISH'),
        'sacrifice': OBSERVED_NODE_LABELS.index('SACRIFICE'),
        'completed': len(OBSERVED_NODE_LABELS)+1,
        'observed': len(OBSERVED_NODE_LABELS)+2,
        'ap': 2,
        'equipped_uses_indices': [detail_start+i*stride+len(encoder.move_use_bins) for i in deterministic],
        'equip_operation': len(RESOURCE_FIELDS)+4+len(ITEM_CATEGORIES)+OPTION_OPERATIONS.index('equip'),
        'deterministic_option_identity_indices': [encoder.option_feature_dim-len(ids)+ids.index(moves[i]) for i in deterministic],
    }


def action_features(tensors, layout):
    """B x (N+O) x 6; a completed flag is a feature, never a hard exclusion.

Random transport exposes an activation at the current node, not its eventual
destination. Its equipped identity is deliberately excluded from deterministic
trigger features. Node legality and current observed node type remain decisive.
    """
    node, glob, option = (tensors[k] for k in ('node_features','global_features','option_features'))
    mask = tensors['action_mask'].bool()
    if node.shape[-1] != layout['node_dim'] or glob.shape[-1] != layout['global_dim'] or option.shape[-1] != layout['option_dim']:
        raise ValueError('adapter feature dimensions differ')
    n = node.shape[1]
    visible = node[:,:,layout['observed']] * tensors['node_mask'].to(node.dtype)
    p05 = glob[:,layout['p05']].unsqueeze(1)
    p06 = glob[:,layout['p06']].unsqueeze(1)
    wish = node[:,:,layout['wish']] * visible * p05
    sac = node[:,:,layout['sacrifice']] * visible * p06 * 2.0
    uses = glob[:,layout['equipped_uses_indices']].sum(1,keepdim=True)
    equipped = (uses > 0).to(node.dtype)
    wish_paid = wish * equipped * mask[:,:n]
    sac_paid = sac * equipped * mask[:,:n]
    total = wish_paid + sac_paid
    result = node.new_zeros((node.shape[0], mask.shape[1], len(FEATURE_NAMES)))
    result[:,:n,0] = wish_paid
    result[:,:n,1] = sac_paid
    result[:,:n,2] = total * node[:,:,layout['completed']]
    result[:,:n,3] = total * glob[:,layout['ap']].unsqueeze(1)
    result[:,:n,4] = total * uses
    # This is visible potential, not a claim that every gear option can reach it.
    # The frozen network still distinguishes range/identity. Its multiplier is
    # learned, and can become zero or negative if this coarse signal is unhelpful.
    potential = (wish+sac).amax(1,keepdim=True)
    candidate = option[:,:,layout['equip_operation']] * option[:,:,layout['deterministic_option_identity_indices']].sum(-1)
    result[:,n:,5] = candidate * potential
    return torch.where(mask.unsqueeze(-1), result, torch.zeros_like(result))


class ConceptResidualNetwork(nn.Module):
    def __init__(self, base, layout):
        super().__init__()
        self.base = base
        self.layout = layout
        self.config = base.config
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
        residual = self.head(action_features(tensors,self.layout)).squeeze(-1)
        adjusted = logits + residual
        adjusted = adjusted.masked_fill(~tensors['action_mask'].bool(), torch.finfo(logits.dtype).min)
        return adjusted, values


class ConceptAdapterController:
    ALGORITHM = 'PUBLIC_CONCEPT_RESIDUAL_DIRECT_NEURAL_V1'

    def __init__(self, base_trainer, base_checkpoint):
        self.base = base_trainer
        self.base_checkpoint = Path(base_checkpoint).resolve()
        self.base_checkpoint_sha256 = sha256(self.base_checkpoint.read_bytes()).hexdigest()
        self.simulator = self.base.simulator
        self.encoder = self.base.encoder
        self.config = self.base.config
        self.policy_constraints = self.base.policy_constraints
        self.layout = feature_layout(self.encoder)
        self.model = ConceptResidualNetwork(self.base.model,self.layout).to(self.config.device)
        self.training_seeds = set(self.base.training_seeds)
        self.learning_history = list(self.base.learning_history)
        self.rng = random.Random(self.config.seed)

    @property
    def implementation_sha256(self):
        return implementation_sha256()

    @property
    def network_implementation_sha256(self):
        return sha256((self.base.network_implementation_sha256+self.implementation_sha256).encode()).hexdigest()

    def _tensors(self, encoded):
        return self.base._tensors(encoded)

    def choose_action(self, state, *, deterministic=False, rng=None):
        encoded = self.encoder.encode(state)
        legal = np.flatnonzero(encoded.action_mask)
        if not len(legal):
            raise RuntimeError('no scope-legal action for concept adapter')
        self.model.eval()
        with torch.inference_mode():
            logits,_ = self.model(**self._tensors([encoded]))
            scores = logits[0,torch.as_tensor(legal,device=self.config.device)]
            if deterministic:
                return int(legal[int(scores.argmax())])
            probabilities = scores.softmax(0).cpu().tolist()
        return int((rng or self.rng).choices(legal.tolist(),weights=probabilities,k=1)[0])

    def save_checkpoint(self,path,*,training=None):
        path=Path(path).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={'algorithm':self.ALGORITHM,'format_version':1,
            'base_checkpoint':str(self.base_checkpoint),'base_checkpoint_sha256':self.base_checkpoint_sha256,
            'environment_sha256':self.simulator.environment_sha256,
            'feature_schema_sha256':self.encoder.schema_sha256,
            'feature_code_sha256':self.implementation_sha256,'feature_layout':self.layout,
            'head_state_dict':self.model.head.state_dict(),
            'training_seeds':sorted(self.training_seeds),'learning_history':self.learning_history,
            'training':training or {},'teacher_used_at_inference':False,
            'action_semantics':'direct legal simulator action; learned residual on frozen direct network'}
        temporary=path.with_name(path.name+'.tmp')
        torch.save(payload,temporary)
        temporary.replace(path)
        return path

    @classmethod
    def load_checkpoint(cls,simulator,path,*,device='cpu'):
        payload=torch.load(Path(path),map_location=device,weights_only=False)
        if payload.get('algorithm')!=cls.ALGORITHM or payload.get('format_version')!=1:
            raise ValueError('not a supported concept adapter checkpoint')
        base_path=Path(payload['base_checkpoint'])
        if sha256(base_path.read_bytes()).hexdigest()!=payload['base_checkpoint_sha256']:
            raise ValueError('base checkpoint bytes differ')
        if payload.get('feature_code_sha256')!=implementation_sha256():
            raise ValueError('adapter feature implementation differs')
        result=cls(PPOTrainer.load_checkpoint(simulator,base_path,device=device),base_path)
        if payload['environment_sha256']!=simulator.environment_sha256 or payload['feature_schema_sha256']!=result.encoder.schema_sha256 or payload['feature_layout']!=result.layout:
            raise ValueError('adapter environment or public feature schema differs')
        result.model.head.load_state_dict(payload['head_state_dict'])
        result.training_seeds.update(payload['training_seeds'])
        if any(PPOTrainer._reserved_seed(s) for s in result.training_seeds):
            raise ValueError('adapter training overlaps reserved evaluation seeds')
        result.learning_history=payload['learning_history']
        result.model.eval()
        return result
