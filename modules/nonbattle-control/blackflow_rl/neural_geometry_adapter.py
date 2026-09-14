"""Learn route/trade decisions from explicit public geometry and item facts.

Distances are topological edge counts, not predicted AP costs or legal future
moves. All inputs derive from already encoded observations and public catalogue
descriptions. No teacher, rollout preview, or hand-authored score runs here.
"""
from hashlib import sha256
from pathlib import Path
import random

import torch
from torch import nn

from . import neural_route_adapter as route
from . import neural_concept_adapter as concept
from .features import OBSERVED_NODE_LABELS,INVENTORY_FLAGS,RESOURCE_FIELDS,ITEM_CATEGORIES,OPTION_OPERATIONS
from .ppo import PPOTrainer

GOALS=('BATTLE_SHOP','SCRAP_SHOP','WISH','SACRIFICE','PORTAL','EXPEDITION','FINAL','BATTLE_ELITE')


def implementation_sha256():
    return sha256(Path(__file__).read_bytes()+route.implementation_sha256().encode()).hexdigest()


def feature_layout(encoder,hidden=64):
    layout=route.feature_layout(encoder)
    identities=encoder.item_identities
    catalogue=encoder.simulator.economy.catalog
    properties=[]
    for identity in identities:
        item=catalogue[identity]
        offsets=item.move_range
        properties.append([float(item.category=='MOVE'),float(item.random_move),
            float(item.expires_on_floor_change),(item.move_ap or 0)/5.,
            (item.move_uses or 0)/5.,max((abs(x) for x,y in offsets),default=0)/5.,
            max((abs(y) for x,y in offsets),default=0)/5.,len(offsets)/20.])
    economy_start=encoder.GLOBAL_BASE_DIM+len(INVENTORY_FLAGS)+encoder.inventory_detail_dim
    operations_start=len(RESOURCE_FIELDS)+4+len(ITEM_CATEGORIES)
    physical_start=encoder.option_feature_dim-len(identities)-12
    option_indices=list(range(operations_start+len(OPTION_OPERATIONS)))+list(range(physical_start,physical_start+12))
    global_indices=list(range(encoder.GLOBAL_BASE_DIM))+list(range(economy_start,economy_start+encoder.ECONOMY_GLOBAL_DIM))
    layout.update(geometry_version=1,hidden=hidden,goals=GOALS,
        goal_indices=[OBSERVED_NODE_LABELS.index(g) for g in GOALS],
        node_local_dim=len(OBSERVED_NODE_LABELS)+encoder.NODE_SCALAR_DIM,
        option_indices=option_indices,global_indices=global_indices,
        item_properties=properties,
        node_valid=len(OBSERVED_NODE_LABELS)+2,
        item_identity_start=encoder.option_feature_dim-len(identities))
    layout['input_dim']=(len(route.FEATURE_NAMES)+layout['node_local_dim']+len(option_indices)
        +len(global_indices)+2+len(GOALS)*5+8)
    return layout


def public_distances(adjacency,node_mask):
    """All-pairs unweighted distances on the public graph, masking padding."""
    b,n,_=adjacency.shape
    valid=node_mask.bool()
    edges=(adjacency>0)&valid[:,:,None]&valid[:,None,:]
    distance=adjacency.new_full((b,n,n),float(n+1))
    distance=torch.where(edges,torch.ones_like(distance),distance)
    eye=torch.eye(n,dtype=torch.bool,device=adjacency.device)[None]
    distance=torch.where(eye&valid[:,:,None],torch.zeros_like(distance),distance)
    for k in range(n):
        distance=torch.minimum(distance,distance[:,:,k:k+1]+distance[:,k:k+1,:])
    return distance


def action_features(tensors,layout):
    node=tensors['node_features']; glob=tensors['global_features']; option=tensors['option_features']
    mask=tensors['action_mask'].bool(); valid=tensors['node_mask'].bool()
    b,n,_=node.shape; actions=mask.shape[1]
    distance=public_distances(tensors['adjacency'],valid)
    # A hidden type must not become a target even if an invalid caller placed
    # a type bit there. Normal FeatureEncoder already emits only coarse labels.
    visible=node[:,:,layout['observed']].bool()&valid
    goals=node[:,:,layout['goal_indices']]*visible.unsqueeze(-1)
    counts=goals.sum(1)/20.
    incomplete=goals*(1-node[:,:,layout['completed']]).unsqueeze(-1)
    incomplete_counts=incomplete.sum(1)/20.
    distances=distance[:,:,:,None].expand(-1,-1,-1,len(GOALS))
    nearest=distances.masked_fill(~goals[:,None,:,:].bool(),float(n+1)).amin(2)
    inverse=torch.where(nearest<=n,1/(1+nearest),torch.zeros_like(nearest))
    current=node[:,:,layout['current']]*valid
    current_distance=(inverse*current.unsqueeze(-1)).sum(1)
    nearby=((distance<=2).to(node.dtype)@incomplete)/10.
    result=node.new_zeros((b,actions,layout['input_dim']))
    offset=0
    first=route.action_features(tensors,layout)
    result[:,:,:first.shape[-1]]=first; offset+=first.shape[-1]
    width=layout['node_local_dim']
    result[:,:n,offset:offset+width]=node[:,:,:width]; offset+=width
    width=len(layout['option_indices'])
    result[:,n:,offset:offset+width]=option[:,:,layout['option_indices']]; offset+=width
    context=torch.cat((glob[:,layout['global_indices']],glob[:,[layout['p05'],layout['p06']]]),dim=1)
    width=context.shape[1]
    result[:,:,offset:offset+width]=context[:,None,:]; offset+=width
    for value in (counts,incomplete_counts,current_distance):
        result[:,:,offset:offset+len(GOALS)]=value[:,None,:]; offset+=len(GOALS)
    result[:,:n,offset:offset+len(GOALS)]=inverse; offset+=len(GOALS)
    result[:,:n,offset:offset+len(GOALS)]=nearby; offset+=len(GOALS)
    properties=node.new_tensor(layout['item_properties'])
    result[:,n:,offset:offset+8]=option[:,:,layout['item_identity_start']:]@properties
    assert offset+8==layout['input_dim']
    # Clamp public scaling outliers; this is input scaling, not reward clipping.
    result=result.clamp(-10,10)
    return torch.where(mask.unsqueeze(-1),result,torch.zeros_like(result))


class GeometryResidualNetwork(nn.Module):
    def __init__(self,base,layout):
        super().__init__()
        self.base=base; self.layout=layout; self.config=base.config
        self.base.requires_grad_(False)
        if layout['hidden']:
            self.head=nn.Sequential(nn.Linear(layout['input_dim'],layout['hidden']),nn.Tanh(),
                nn.Linear(layout['hidden'],1,bias=False))
            nn.init.zeros_(self.head[-1].weight)
        else:
            self.head=nn.Linear(layout['input_dim'],1,bias=False)
            nn.init.zeros_(self.head.weight)

    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self

    def forward(self,**tensors):
        with torch.no_grad():
            logits,value=self.base(**tensors)
        logits=logits+self.head(action_features(tensors,self.layout)).squeeze(-1)
        return logits.masked_fill(~tensors['action_mask'].bool(),torch.finfo(logits.dtype).min),value


class GeometryAdapterController(concept.ConceptAdapterController):
    ALGORITHM='PUBLIC_GEOMETRY_RESIDUAL_DIRECT_NEURAL_V1'

    def __init__(self,base_trainer,base_checkpoint,hidden=64):
        self.base=base_trainer
        self.base_checkpoint=Path(base_checkpoint).resolve()
        self.base_checkpoint_sha256=sha256(self.base_checkpoint.read_bytes()).hexdigest()
        self.simulator,self.encoder,self.config=base_trainer.simulator,base_trainer.encoder,base_trainer.config
        self.policy_constraints=base_trainer.policy_constraints
        self.layout=feature_layout(self.encoder,hidden)
        self.model=GeometryResidualNetwork(base_trainer.model,self.layout).to(self.config.device)
        self.training_seeds=set(base_trainer.training_seeds)
        self.learning_history=list(base_trainer.learning_history)
        self.rng=random.Random(self.config.seed)

    @property
    def implementation_sha256(self):
        return implementation_sha256()

    @classmethod
    def load_checkpoint(cls,simulator,path,*,device='cpu'):
        payload=torch.load(path,map_location=device,weights_only=False)
        if payload.get('algorithm')!=cls.ALGORITHM or payload.get('format_version')!=1:
            raise ValueError('unsupported geometry adapter')
        base=Path(payload['base_checkpoint'])
        if sha256(base.read_bytes()).hexdigest()!=payload['base_checkpoint_sha256']:
            raise ValueError('base checkpoint bytes differ')
        if payload['feature_code_sha256']!=implementation_sha256():
            raise ValueError('geometry feature implementation differs')
        result=cls(PPOTrainer.load_checkpoint(simulator,base,device=device),base,payload['feature_layout']['hidden'])
        if (payload['environment_sha256']!=simulator.environment_sha256
            or payload['feature_schema_sha256']!=result.encoder.schema_sha256
            or payload['feature_layout']!=result.layout):
            raise ValueError('geometry environment or input layout differs')
        result.model.head.load_state_dict(payload['head_state_dict'])
        result.training_seeds.update(payload['training_seeds'])
        if any(PPOTrainer._reserved_seed(s) for s in result.training_seeds):
            raise ValueError('training overlaps reserved evaluation seeds')
        result.learning_history=payload['learning_history']; result.model.eval()
        return result
