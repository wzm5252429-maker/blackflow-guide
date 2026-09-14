"""Candidate-to-public-node attention over frozen source features.

This layer reads the currently encoded board. Relative coordinates describe
observed geometry, not future transitions, reachability promises or a search.
"""
from hashlib import sha256
from pathlib import Path
import math
import torch
from torch import nn
from .features import OBSERVED_NODE_LABELS,FeatureEncoder,SHOP_SCALAR_FIELDS,SHOP_CATEGORIES
from .neural_macro_route import MacroRouteController
from .neural_mobility_route import MobilityResidualNetwork,MobilityRouteController,NAMES
from .neural_context_mobility import ContextMobilityHead,ContextMobilityController,implementation_sha256 as context_sha

NODE_OFFSET=len(OBSERVED_NODE_LABELS)
PUBLIC_NODE_DIM=NODE_OFFSET+FeatureEncoder.NODE_SCALAR_DIM+len(SHOP_SCALAR_FIELDS)+2*len(SHOP_CATEGORIES)
PUBLIC_GLOBAL_DIM=FeatureEncoder.GLOBAL_BASE_DIM


def implementation_sha256():
    return sha256(Path(__file__).read_bytes()+context_sha().encode()).hexdigest()


def capture_public_source(base,tensors):
    context=[]; nodes=[]; handles=[]
    try:
        handles.append(base.head[1].register_forward_hook(lambda module,args,value:context.append(value.detach())))
        if not len(base.backbone.message_passing): raise ValueError('Expected an existing source graph representation')
        handles.append(base.backbone.message_passing[-1].register_forward_hook(lambda module,args,value:nodes.append(value.detach())))
        with torch.no_grad():
            scores=base(**tensors); gears=base.gear_embedding(tensors['macro_gears']).detach()
        if len(context)!=1 or len(nodes)!=1 or context[0].shape[:2]!=scores.shape or nodes[0].shape[:2]!=tensors['node_mask'].shape:
            raise ValueError('Frozen source attention features are misaligned')
        return scores,context[0],nodes[0],gears
    finally:
        for handle in handles: handle.remove()


class PublicNodeAttentionHead(nn.Module):
    def __init__(self,source_dim=128,gear_dim=32,hidden=64,attention_dim=32,attention_enabled=True):
        super().__init__(); self.source_dim=source_dim; self.gear_dim=gear_dim
        self.attention_dim=attention_dim; self.attention_enabled=attention_enabled
        self.context_norm=nn.LayerNorm(source_dim); self.node_norm=nn.LayerNorm(source_dim)
        query_dim=source_dim+len(NAMES)+gear_dim+PUBLIC_GLOBAL_DIM
        node_dim=source_dim+PUBLIC_NODE_DIM
        self.query=nn.Linear(query_dim,attention_dim)
        self.key=nn.Linear(node_dim,attention_dim,bias=False)
        self.relative_key=nn.Linear(4,attention_dim,bias=False)
        self.value=nn.Sequential(nn.Linear(node_dim,attention_dim),nn.GELU())
        self.relative_value=nn.Linear(4,attention_dim,bias=False)
        self.output=nn.Sequential(nn.Linear(query_dim+2*attention_dim,hidden),nn.Tanh(),nn.Linear(hidden,1,bias=False))
        nn.init.zeros_(self.output[-1].weight)

    def forward(self,*,context,mobility,source_nodes,public_nodes,node_mask,macro_nodes,gear_features,public_global):
        width=context.shape[1]
        query_features=torch.cat((self.context_norm(context),mobility,gear_features,
            public_global[:,None,:].expand(-1,width,-1)),-1)
        if not self.attention_enabled:
            empty=query_features.new_zeros((*query_features.shape[:2],2*self.attention_dim))
            return self.output(torch.cat((query_features,empty),-1)).squeeze(-1)
        mask=node_mask.bool(); clean=torch.where(mask.unsqueeze(-1),public_nodes,0.)
        node_features=torch.cat((self.node_norm(torch.where(mask.unsqueeze(-1),source_nodes,0.)),clean),-1)
        # Absolute coordinates are used only to form relative geometry in this
        # new node branch; the unchanged source representation is left intact.
        node_features=node_features.clone()
        node_features[...,self.source_dim+NODE_OFFSET+6:self.source_dim+NODE_OFFSET+8]=0.
        positions=clean[...,NODE_OFFSET+6:NODE_OFFSET+8]
        current=(clean[...,NODE_OFFSET]>0.5).to(torch.int64).argmax(-1)
        batch=torch.arange(context.shape[0],device=context.device)[:,None]
        deterministic_move=(mobility[...,0]>0.5)&(mobility[...,4]<0.5)
        anchors=torch.where(deterministic_move,macro_nodes,current[:,None])
        origins=positions[batch,anchors]
        delta=positions[:,None,:,:]-origins[:,:,None,:]
        # A random landing has no known geometric anchor. It can still read
        # the public board as a whole, exactly as the original global context.
        delta=torch.where((mobility[...,4]<0.5)[...,None,None],delta,0.)
        relative=torch.cat((delta,delta.abs()),-1)
        query=self.query(query_features)
        keys=self.key(node_features)[:,None,:,:]+self.relative_key(relative)
        logits=(query[:,:,None,:]*keys).sum(-1)/math.sqrt(query.shape[-1])
        logits=logits.masked_fill(~mask[:,None,:],torch.finfo(logits.dtype).min)
        weights=logits.softmax(-1)*mask[:,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1.)
        values=self.value(node_features)[:,None,:,:]+self.relative_value(relative)
        pooled=(weights.unsqueeze(-1)*values).sum(-2)
        maximum=values.masked_fill(~mask[:,None,:,None],torch.finfo(values.dtype).min).max(-2).values
        maximum=torch.where(mask.any(-1)[:,None,None],maximum,0.)
        return self.output(torch.cat((query_features,pooled,maximum),-1)).squeeze(-1)


class AttentiveRouteNetwork(MobilityResidualNetwork):
    def __init__(self,base,hidden=64,attention_dim=32,prior=None,attention_enabled=True):
        super().__init__(base,hidden); self.context_dim=base.head[0].out_features
        self.prior=prior
        if self.prior is not None: self.prior.requires_grad_(False); self.prior.eval()
        self.head=PublicNodeAttentionHead(self.context_dim,base.gear_embedding.embedding_dim,hidden,attention_dim,attention_enabled)

    def forward(self,*,mobility=None,**tensors):
        scores,context,nodes,gears=capture_public_source(self.base,tensors)
        mobility=self.inference_mobility if mobility is None else mobility
        if mobility is None or mobility.shape[:2]!=scores.shape: raise ValueError('Missing aligned mobility facts')
        if self.prior is not None:
            with torch.no_grad(): scores=scores+self.prior(torch.cat((context,mobility),-1)).squeeze(-1)
        residual=self.head(context=context,mobility=mobility,source_nodes=nodes,
            public_nodes=tensors['node_features'][...,:PUBLIC_NODE_DIM],node_mask=tensors['node_mask'],
            macro_nodes=tensors['macro_nodes'],gear_features=gears,
            public_global=tensors['global_features'][...,:PUBLIC_GLOBAL_DIM])
        return (scores+residual).masked_fill(~tensors['macro_mask'],torch.finfo(scores.dtype).min)


class AttentiveRouteController(MobilityRouteController):
    ALGORITHM='PUBLIC_CANDIDATE_NODE_ATTENTION_NEURAL_V1'

    def __init__(self,source,source_path,hidden=64,attention_dim=32,prior_path=None,attention_enabled=True):
        super().__init__(source,source_path,hidden); self.attention_dim=attention_dim
        self.prior_path=Path(prior_path).resolve() if prior_path is not None else None
        self.attention_enabled=attention_enabled; prior=None
        if self.prior_path is not None:
            payload=torch.load(self.prior_path,map_location='cpu',weights_only=False)
            if payload['algorithm']!=ContextMobilityController.ALGORITHM or payload['implementation_sha256']!=context_sha() or payload['source_sha256']!=sha256(self.source_path.read_bytes()).hexdigest():
                raise ValueError('Frozen preceding residual changed')
            if payload['environment_sha256']!=self.simulator.environment_sha256 or payload['feature_schema_sha256']!=self.encoder.schema_sha256 or payload['context_dim']!=source.route_model.head[0].out_features:
                raise ValueError('Preceding residual environment or context differs')
            prior=ContextMobilityHead(payload['context_dim'],payload['hidden']); prior.load_state_dict(payload['head_state_dict'])
            self.training_seeds.update(payload['training_seeds']); self.history=list(payload['history'])
        self.route_model=AttentiveRouteNetwork(source.route_model,hidden,attention_dim,prior,attention_enabled)

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix('.tmp')
        torch.save({'algorithm':self.ALGORITHM,'implementation_sha256':implementation_sha256(),
            'source_path':str(self.source_path),'source_sha256':sha256(self.source_path.read_bytes()).hexdigest(),
            'environment_sha256':self.simulator.environment_sha256,'feature_schema_sha256':self.encoder.schema_sha256,
            'hidden':self.hidden,'attention_dim':self.attention_dim,'context_dim':self.route_model.context_dim,
            'attention_enabled':self.attention_enabled,'prior_path':str(self.prior_path) if self.prior_path is not None else None,
            'prior_sha256':sha256(self.prior_path.read_bytes()).hexdigest() if self.prior_path is not None else None,
            'head_state_dict':self.route_model.head.state_dict(),'training_seeds':sorted(self.training_seeds),'history':self.history},temporary)
        temporary.replace(path); return path

    @classmethod
    def load(cls,sim,path):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['algorithm']!=cls.ALGORITHM or payload['implementation_sha256']!=implementation_sha256():
            raise ValueError('Public attention implementation mismatch')
        source=Path(payload['source_path'])
        if sha256(source.read_bytes()).hexdigest()!=payload['source_sha256']: raise ValueError('Frozen source changed')
        if payload['prior_path'] is not None and sha256(Path(payload['prior_path']).read_bytes()).hexdigest()!=payload['prior_sha256']:
            raise ValueError('Frozen preceding attention baseline changed')
        result=cls(MacroRouteController.load(sim,source),source,payload['hidden'],payload['attention_dim'],payload['prior_path'],payload['attention_enabled'])
        if sim.environment_sha256!=payload['environment_sha256'] or result.encoder.schema_sha256!=payload['feature_schema_sha256'] or result.route_model.context_dim!=payload['context_dim']:
            raise ValueError('Attention schema or environment mismatch')
        result.route_model.head.load_state_dict(payload['head_state_dict']); result.training_seeds.update(payload['training_seeds'])
        result.history=payload['history']; result.route_model.eval(); return result
