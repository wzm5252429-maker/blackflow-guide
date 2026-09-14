"""Joint public equipment/destination policy; no teacher/search at inference.

Candidates enumerate only the current legal movement menu under each publicly
available equipment setting. Changing that UI is not a game rollout. Random
vehicles remain one activation, never a choice of a hidden landing/reward.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .domain import ActionKind
from .neural_memory_v2 import FairEquipmentCycleMemory, public_progress_key
from .policy_constraints import allowed_action_ids


FACTS = ('move', 'menu', 'gear', 'same_equipment', 'random', 'movement_cost_over_5',
         'uses_over_10', 'expires', 'delta_row_over_10', 'delta_col_over_10',
         'known_wish_copies_over_10', 'known_sacrifice_relics_over_10',
         'completed', 'revealed', 'p05_over_10', 'p06_over_10', 'floor_over_5',
         'ap_over_20', 'vehicle_count_over_10', 'remaining_uses_over_20')


def implementation_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def route_state(state):
    return (state.resources.parts <= state.parts_capacity and
            (not state.pending_node_id or state.floor_map.node(state.pending_node_id).node_type.value in ('FINAL', 'EVACUATE')))


@dataclass(frozen=True)
class Macro:
    first_action: int
    target: str | None
    gear: str | None
    semantic: tuple
    node_index: int
    option_index: int
    facts: tuple
    gear_type: int
    requires_equip: bool


def candidates(sim, encoder, state):
    """Never read private node types, future floors, or random outcomes."""
    state = sim.belief_state(state)
    permitted = set(allowed_action_ids(sim, state, encoder.policy_constraints))
    legal = [a for a in sim.legal_actions(state) if a.action_id in permitted]
    gear_ids = {identity: i+1 for i, identity in enumerate(encoder.move_identities)}
    held = {x.instance_id: x for x in state.item_instances}
    p05 = sum(x.item_id.endswith('P_05') for x in held.values())
    p06 = sum(x.item_id.endswith('P_06') for x in held.values())
    vehicles = [x for x in held.values() if x.category == 'MOVE']
    current = state.floor_map.node(state.current_node_id)
    result = []

    def move_candidate(action, gear_id, first_action, requires_equip):
        gear = held.get(gear_id)
        definition = sim.economy.catalog[gear.item_id] if gear else None
        random_move = bool(definition and definition.random_move)
        node = state.floor_map.node(action.target_node_id)
        known = node.node_id in state.revealed and not random_move
        facts = (1., 0., float(gear is not None), float(gear_id == state.equipped_instance_id),
            float(random_move), action.movement_cost/5., (gear.uses_remaining or 0)/10. if gear else 0.,
            float(bool(definition and definition.expires_on_floor_change)),
            (node.row-current.row)/10. if not random_move else 0.,
            (node.col-current.col)/10. if not random_move else 0.,
            p05/10. if known and node.node_type.value == 'WISH' and gear else 0.,
            2*p06/10. if known and node.node_type.value == 'SACRIFICE' and gear else 0.,
            float(node.node_id in state.completed and not random_move), float(known),
            p05/10., p06/10., state.floor/5., state.resources.action_points/20.,
            len(vehicles)/10., sum(x.uses_remaining or 0 for x in vehicles)/20.)
        # Same public item type/uses and landing are interchangeable labels;
        # instance identity is used for execution only, never an NN feature.
        semantic = ('MOVE', gear.item_id if gear else None, gear.uses_remaining if gear else None,
                    None if random_move else node.node_id)
        result.append(Macro(first_action, node.node_id, gear_id, semantic,
            -1 if random_move else node.index, -1, facts,
            gear_ids[gear.item_id] if gear else 0, requires_equip))

    for action in legal:
        if action.kind == ActionKind.MOVE:
            move_candidate(action, state.equipped_instance_id, action.action_id, False)
        elif action.kind == ActionKind.EQUIP:
            view = replace(state, equipped_instance_id=action.equipment_instance_id)
            permitted_after = set(allowed_action_ids(sim, view, encoder.policy_constraints))
            for onward in sim.legal_actions(view):
                if onward.kind == ActionKind.MOVE and onward.action_id in permitted_after:
                    move_candidate(onward, action.equipment_instance_id, action.action_id, True)
        else:
            facts = (0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     p05/10., p06/10., state.floor/5., state.resources.action_points/20.,
                     len(vehicles)/10., sum(x.uses_remaining or 0 for x in vehicles)/20.)
            result.append(Macro(action.action_id, None, None, ('MENU', action.action_id),
                                -1, action.option_index, facts, 0, False))
    return result


def teacher_targets(macros, action, commitment):
    choices = [i for i, m in enumerate(macros) if m.first_action == action]
    if commitment is not None:
        _, gear, target = commitment
        choices = [i for i in choices if macros[i].gear == gear and
                   (target is None or macros[i].target == target)]
    if not choices:
        raise ValueError('Teacher action/intent has no legal public macro')
    semantics = {macros[i].semantic for i in choices}
    return [i for i, m in enumerate(macros) if m.semantic in semantics]


def macro_tensors(groups):
    width = max(len(g) for g in groups)
    facts = np.zeros((len(groups), width, len(FACTS)), dtype=np.float32)
    nodes = np.zeros((len(groups), width), dtype=np.int64)
    options = nodes.copy(); gears = nodes.copy()
    mask = np.zeros((len(groups), width), dtype=bool)
    for b, group in enumerate(groups):
        for i, m in enumerate(group):
            facts[b, i] = m.facts
            nodes[b, i] = max(0, m.node_index)
            options[b, i] = max(0, m.option_index)
            gears[b, i] = m.gear_type
            mask[b, i] = True
    return {'macro_facts': torch.from_numpy(facts), 'macro_nodes': torch.from_numpy(nodes),
            'macro_options': torch.from_numpy(options), 'macro_gears': torch.from_numpy(gears),
            'macro_mask': torch.from_numpy(mask)}


class MacroRouteNetwork(nn.Module):
    def __init__(self, base, gear_types):
        super().__init__()
        self.backbone = deepcopy(base)
        self.backbone.value_head.requires_grad_(False)
        width = base.config.hidden_dim
        self.gear_embedding = nn.Embedding(gear_types+1, 32)
        self.head = nn.Sequential(nn.Linear(width*4+32+len(FACTS), width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, *, macro_facts, macro_nodes, macro_options, macro_gears, macro_mask, **tensors):
        base = self.backbone
        node_mask = tensors['node_mask'].bool()
        option_mask = tensors['option_observation_mask'].bool()
        clean_nodes = torch.where(node_mask.unsqueeze(-1), tensors['node_features'], 0.)
        clean_options = torch.where(option_mask.unsqueeze(-1), tensors['option_features'], 0.)
        valid_edges = node_mask[:, :, None] & node_mask[:, None, :]
        adjacency = torch.where(valid_edges, tensors['adjacency'], 0.)
        hidden = torch.where(node_mask.unsqueeze(-1), base.node_encoder(clean_nodes), 0.)
        outgoing = base._normalize_adjacency(adjacency)
        incoming = base._normalize_adjacency(adjacency.transpose(1, 2))
        for layer in base.message_passing:
            hidden = layer(hidden, outgoing, incoming, node_mask)
        graph = base._masked_mean(hidden, node_mask)
        glob = base.global_encoder(tensors['global_features'])
        option_hidden = base.option_encoder(clean_options)
        menu = base.menu_encoder(torch.cat((base._masked_mean(option_hidden, option_mask),
                                            base._masked_max(option_hidden, option_mask)), -1))
        menu = torch.where(option_mask.any(1, keepdim=True), menu, 0.)
        b = torch.arange(hidden.shape[0])[:, None]
        # A random activation has no known destination embedding.
        move = macro_facts[:, :, 0:1]
        random_move = macro_facts[:, :, 4:5]
        local = hidden[b, macro_nodes] * move * (1-random_move)
        local = local + option_hidden[b, macro_options] * (1-move)
        width = macro_mask.shape[1]
        context = torch.cat((local, graph[:, None].expand(-1, width, -1),
                             glob[:, None].expand(-1, width, -1), menu[:, None].expand(-1, width, -1)), -1)
        prior = base.node_policy_head(context).squeeze(-1)*move.squeeze(-1)
        prior = prior + base.option_policy_head(context).squeeze(-1)*(1-move.squeeze(-1))
        residual = self.head(torch.cat((context, self.gear_embedding(macro_gears), macro_facts), -1)).squeeze(-1)
        return (prior + residual).masked_fill(~macro_mask, torch.finfo(prior.dtype).min)


class MacroRouteController:
    ALGORITHM = 'PUBLIC_JOINT_EQUIPMENT_DESTINATION_NEURAL_V1'

    def __init__(self, base, base_path):
        self.base_path = Path(base_path).resolve()
        self.menu = FairEquipmentCycleMemory(base)
        self.simulator, self.encoder, self.config = base.simulator, base.encoder, base.config
        self.policy_constraints = base.policy_constraints
        self.route_model = MacroRouteNetwork(base.model, len(self.encoder.move_identities))
        self.training_seeds = set(base.training_seeds)
        self.history = []
        self.begin_episode()

    def begin_episode(self):
        self.menu.begin_episode()
        self.commitment = None
        self.macro_decisions = self.committed_moves = self.stale_commitments = 0

    def choose_action(self, state, *, deterministic=True, rng=None):
        if not deterministic:
            raise ValueError('This experiment uses deterministic joint-plan inference')
        if self.commitment is not None:
            progress, gear, target = self.commitment
            self.commitment = None
            allowed = set(allowed_action_ids(self.simulator, state, self.policy_constraints))
            action = next((a for a in self.simulator.legal_actions(state) if a.action_id in allowed and
                           a.kind == ActionKind.MOVE and a.target_node_id == target and
                           a.equipment_instance_id == gear), None)
            if progress == public_progress_key(state) and state.equipped_instance_id == gear and action is not None:
                self.committed_moves += 1
                return int(action.action_id)
            self.stale_commitments += 1
        if not route_state(state):
            return self.menu.choose_action(state, deterministic=True, rng=rng)
        macros = candidates(self.simulator, self.encoder, state)
        if not macros:
            return self.menu.choose_action(state, deterministic=True, rng=rng)
        self.route_model.eval()
        with torch.inference_mode():
            inputs = self.menu.trainer._tensors([self.encoder.encode(state)])
            logits = self.route_model(**inputs, **macro_tensors([macros]))
            selected = macros[int(logits[0].argmax())]
        self.menu.begin_episode()
        self.macro_decisions += 1
        if selected.requires_equip:
            self.commitment = public_progress_key(state), selected.gear, selected.target
        return int(selected.first_action)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save({'algorithm': self.ALGORITHM, 'implementation_sha256': implementation_sha256(),
                    'base_path': str(self.base_path), 'base_sha256': sha256(self.base_path.read_bytes()).hexdigest(),
                    'environment_sha256': self.simulator.environment_sha256,
                    'feature_schema_sha256': self.encoder.schema_sha256,
                    'route_state_dict': self.route_model.state_dict(),
                    'training_seeds': sorted(self.training_seeds), 'history': self.history}, temporary)
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, sim, path):
        from .ppo import PPOTrainer
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['algorithm'] != cls.ALGORITHM or payload['implementation_sha256'] != implementation_sha256():
            raise ValueError('Macro controller version/implementation mismatch')
        base = Path(payload['base_path'])
        if sha256(base.read_bytes()).hexdigest() != payload['base_sha256']:
            raise ValueError('Frozen menu checkpoint differs')
        result = cls(PPOTrainer.load_checkpoint(sim, base), base)
        if result.encoder.schema_sha256 != payload['feature_schema_sha256'] or sim.environment_sha256 != payload['environment_sha256']:
            raise ValueError('Macro environment/features differ')
        result.route_model.load_state_dict(payload['route_state_dict'])
        result.training_seeds.update(payload['training_seeds'])
        result.history = payload['history']
        result.route_model.eval()
        return result
