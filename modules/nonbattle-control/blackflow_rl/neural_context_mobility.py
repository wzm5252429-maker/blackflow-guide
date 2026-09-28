"""Frozen source context plus public mobility, learned by a small residual."""
from hashlib import sha256
from pathlib import Path
import torch
from torch import nn

from .neural_macro_route import MacroRouteController
from .neural_mobility_route import (MobilityResidualNetwork,MobilityRouteController,NAMES,
                                    implementation_sha256 as mobility_sha)


def implementation_sha256():
    return sha256(Path(__file__).read_bytes()+mobility_sha().encode()).hexdigest()


def capture_context(base,tensors):
    """Read an existing frozen activation without changing source arithmetic."""
    captured=[]; handle=base.head[1].register_forward_hook(lambda module,args,output:captured.append(output.detach()))
    try:
        with torch.no_grad(): scores=base(**tensors)
        if len(captured)!=1 or captured[0].shape[:2]!=scores.shape: raise ValueError('Source context layout differs')
        return scores,captured[0]
    finally: handle.remove()


class ContextMobilityHead(nn.Module):
    def __init__(self,context_dim,hidden=64):
        super().__init__(); self.context_dim=context_dim
        self.norm=nn.LayerNorm(context_dim)
        self.layers=nn.Sequential(nn.Linear(context_dim+len(NAMES),hidden),nn.Tanh(),nn.Linear(hidden,1,bias=False))
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self,features):
        context=self.norm(features[...,:self.context_dim]); facts=features[...,self.context_dim:]
        return self.layers(torch.cat((context,facts),-1))


class ContextMobilityNetwork(MobilityResidualNetwork):
    def __init__(self,base,hidden=64):
        super().__init__(base,hidden); self.context_dim=base.head[0].out_features
        self.head=ContextMobilityHead(self.context_dim,hidden)

    def forward(self,*,mobility=None,**tensors):
        logits,context=capture_context(self.base,tensors)
        mobility=self.inference_mobility if mobility is None else mobility
        if mobility is None or mobility.shape[:2]!=logits.shape: raise ValueError('Missing public mobility facts')
        return (logits+self.head(torch.cat((context,mobility),-1)).squeeze(-1)).masked_fill(~tensors['macro_mask'],torch.finfo(logits.dtype).min)


class ContextMobilityController(MobilityRouteController):
    ALGORITHM='PUBLIC_CONTEXTUAL_MOBILITY_RESIDUAL_NEURAL_V1'

    def __init__(self,source,source_path,hidden=64):
        super().__init__(source,source_path,hidden); self.route_model=ContextMobilityNetwork(source.route_model,hidden)

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix('.tmp')
        torch.save({'algorithm':self.ALGORITHM,'implementation_sha256':implementation_sha256(),
            'source_path':str(self.source_path),'source_sha256':sha256(self.source_path.read_bytes()).hexdigest(),
            'environment_sha256':self.simulator.environment_sha256,'feature_schema_sha256':self.encoder.schema_sha256,
            'hidden':self.hidden,'context_dim':self.route_model.context_dim,'head_state_dict':self.route_model.head.state_dict(),
            'training_seeds':sorted(self.training_seeds),'history':self.history},temporary)
        temporary.replace(path); return path

    @classmethod
    def load(cls,sim,path):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['algorithm']!=cls.ALGORITHM or payload['implementation_sha256']!=implementation_sha256(): raise ValueError('Context mobility implementation mismatch')
        source=Path(payload['source_path'])
        if sha256(source.read_bytes()).hexdigest()!=payload['source_sha256']: raise ValueError('Frozen source changed')
        result=cls(MacroRouteController.load(sim,source),source,payload['hidden'])
        if sim.environment_sha256!=payload['environment_sha256'] or result.encoder.schema_sha256!=payload['feature_schema_sha256'] or result.route_model.context_dim!=payload['context_dim']:
            raise ValueError('Context mobility schema/environment mismatch')
        result.route_model.head.load_state_dict(payload['head_state_dict']); result.training_seeds.update(payload['training_seeds'])
        result.history=payload['history']; result.route_model.eval(); return result
