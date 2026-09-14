"""Small joint-route residual using explicit public vehicle connectivity.

Descriptors inspect a fixed observed board and known movement rules. They do
not execute transitions, reveal nodes, call a teacher, or score a future plan.
Connections describe static geometry, not a promise that a future event or
moving board will remain unchanged.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .neural_macro_route import MacroRouteController,candidates,route_state,FACTS,implementation_sha256 as macro_sha


GOALS=('WISH','SACRIFICE','BATTLE_SHOP','SCRAP_SHOP','PORTAL','EXPEDITION','FINAL','BATTLE_ELITE')
EXTRA=('ap_after','same_gear_uses_after','other_vehicle_uses','random_vehicle_uses',
       'dynamic_blue_board','resident_board','has_future_vehicle','future_p05_possible','future_p06_possible')
NAMES=FACTS+tuple('target_'+g for g in GOALS)+EXTRA+tuple('same_gear_'+g for g in GOALS)+tuple('any_gear_'+g for g in GOALS)+tuple('new_any_gear_'+g for g in GOALS)


def implementation_sha256():
    return sha256(Path(__file__).read_bytes()+macro_sha().encode()).hexdigest()


def mobility_facts(sim,encoder,state,macros):
    """No simulator transition: read catalogue movement masks on public copies."""
    state=sim.belief_state(state)
    held={x.instance_id:x for x in state.item_instances}
    known={n.node_id:n for n in state.floor_map.nodes if n.node_id in state.revealed}
    goal_sets={g:{n.node_id for n in known.values() if n.node_type.value==g} for g in GOALS}
    p05=sum(x.item_id.endswith('P_05') for x in held.values())
    p06=sum(x.item_id.endswith('P_06') for x in held.values())
    dynamic=bool(state.portal_context and state.portal_context.variation_id==5)
    residents=bool(state.resident_context)
    result=[]
    for macro in macros:
        target=known.get(macro.target)
        target_bits=[float(bool(target and target.node_type.value==g and not macro.facts[4])) for g in GOALS]
        same=set(); reachable=set(); other_uses=random_uses=same_after=0; future=False
        ap_after=max(0.,state.resources.action_points-macro.facts[5]*5.)
        # A random activation has no publicly known landing. A menu choice has
        # no implied movement. Re-rolling blue boards cannot hold node types fixed.
        if macro.facts[0] and not macro.facts[4] and macro.target is not None and not dynamic:
            vehicles=[]
            for item in held.values():
                if item.category!='MOVE': continue
                uses=(item.uses_remaining or 0)-int(item.instance_id==macro.gear)
                if uses<=0: continue
                item=replace(item,uses_remaining=uses); definition=sim.economy.catalog[item.item_id]
                if item.instance_id==macro.gear: same_after=uses
                else: other_uses+=uses
                if definition.random_move:
                    random_uses+=uses; continue
                vehicles.append(item)
            public_items=tuple(replace(item,uses_remaining=(item.uses_remaining or 0)-1)
                               if item.instance_id==macro.gear else item for item in state.item_instances)
            moved_view=replace(state,current_node_id=macro.target,pending_node_id=None,item_instances=public_items,
                resources=replace(state.resources,action_points=int(round(ap_after))))
            for vehicle in vehicles:
                view=replace(moved_view,equipped_instance_id=vehicle.instance_id)
                connections={node for node,cost in sim.economy.movement_targets(view,{}).items() if cost<=ap_after}
                future=future or bool(connections)
                reachable.update(connections)
                if vehicle.instance_id==macro.gear: same.update(connections)
        extra=[ap_after/20.,same_after/10.,other_uses/20.,random_uses/20.,float(dynamic),float(residents),float(future),
               p05/10. if reachable&goal_sets['WISH'] else 0.,2*p06/10. if reachable&goal_sets['SACRIFICE'] else 0.]
        row=list(macro.facts)+target_bits+extra
        row += [len(same&goal_sets[g])/10. for g in GOALS]
        row += [len(reachable&goal_sets[g])/10. for g in GOALS]
        row += [len((reachable&goal_sets[g])-state.completed)/10. for g in GOALS]
        if len(row)!=len(NAMES): raise ValueError('Mobility feature layout differs')
        result.append(row)
    return np.asarray(result,dtype=np.float32)


def mobility_tensors(rows):
    width=max(len(r['macros']) for r in rows)
    array=np.zeros((len(rows),width,len(NAMES)),dtype=np.float32)
    for i,row in enumerate(rows):
        if np.asarray(row['mobility']).shape!=(len(row['macros']),len(NAMES)): raise ValueError('Bad mobility data shape')
        array[i,:len(row['macros'])]=row['mobility']
    return {'mobility':torch.from_numpy(array)}


class MobilityResidualNetwork(nn.Module):
    def __init__(self,base,hidden=64):
        super().__init__(); self.base=base; self.base.requires_grad_(False); self.hidden=hidden
        self.head=nn.Sequential(nn.Linear(len(NAMES),hidden),nn.Tanh(),nn.Linear(hidden,1,bias=False))
        nn.init.zeros_(self.head[-1].weight); self.inference_mobility=None

    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self

    def forward(self,*,mobility=None,**tensors):
        with torch.no_grad(): logits=self.base(**tensors)
        mobility=self.inference_mobility if mobility is None else mobility
        if mobility is None or mobility.shape[:2]!=logits.shape: raise ValueError('Missing/misaligned public mobility facts')
        return (logits+self.head(mobility).squeeze(-1)).masked_fill(~tensors['macro_mask'],torch.finfo(logits.dtype).min)


class MobilityRouteController(MacroRouteController):
    ALGORITHM='PUBLIC_JOINT_MOBILITY_RESIDUAL_NEURAL_V1'

    def __init__(self,source,source_path,hidden=64):
        super().__init__(source.menu.trainer,source.base_path)
        self.source_path=Path(source_path).resolve(); self.hidden=hidden
        self.route_model=MobilityResidualNetwork(source.route_model,hidden)
        self.training_seeds=set(source.training_seeds); self.history=list(source.history)

    def choose_action(self,state,*,deterministic=True,rng=None):
        # The parent may discard a stale commitment and make a new decision.
        if route_state(state):
            group=candidates(self.simulator,self.encoder,state)
            if group:
                self.route_model.inference_mobility=torch.from_numpy(mobility_facts(self.simulator,self.encoder,state,group))[None]
        try: return super().choose_action(state,deterministic=deterministic,rng=rng)
        finally: self.route_model.inference_mobility=None

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
        torch.save({'algorithm':self.ALGORITHM,'implementation_sha256':implementation_sha256(),
            'source_path':str(self.source_path),'source_sha256':sha256(self.source_path.read_bytes()).hexdigest(),
            'environment_sha256':self.simulator.environment_sha256,'feature_schema_sha256':self.encoder.schema_sha256,
            'hidden':self.hidden,'head_state_dict':self.route_model.head.state_dict(),
            'training_seeds':sorted(self.training_seeds),'history':self.history},temp)
        temp.replace(path); return path

    @classmethod
    def load(cls,sim,path):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['algorithm']!=cls.ALGORITHM or payload['implementation_sha256']!=implementation_sha256():
            raise ValueError('Mobility controller implementation mismatch')
        source=Path(payload['source_path'])
        if sha256(source.read_bytes()).hexdigest()!=payload['source_sha256']: raise ValueError('Frozen source changed')
        result=cls(MacroRouteController.load(sim,source),source,payload['hidden'])
        if sim.environment_sha256!=payload['environment_sha256'] or result.encoder.schema_sha256!=payload['feature_schema_sha256']:
            raise ValueError('Mobility environment/features differ')
        result.route_model.head.load_state_dict(payload['head_state_dict']); result.training_seeds.update(payload['training_seeds'])
        result.history=payload['history']; result.route_model.eval(); return result
