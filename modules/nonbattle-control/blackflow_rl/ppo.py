"""On-policy neural control of the actual simulator, without a hand policy.

Only observed features enter the network. Rollouts execute sampled network
actions in the real episode; no determinized hidden future or beam fallback
supplies actions or policy targets. Old suggested strategies remain separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from .domain import GameState
from .features import FeatureEncoder, stack_encoded
from .policy_constraints import (AUTONOMOUS_POLICY_CONSTRAINTS,
    PolicyConstrainedEnvironment, scope_violations)


def _torch():
    import torch
    return torch


@dataclass(frozen=True, slots=True)
class PPOConfig:
    seed: int = 20260909
    training_seed_start: int = 1_200_000
    num_envs: int = 8
    rollout_steps: int = 128
    minibatch_size: int = 64
    epochs: int = 4
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gamma: float = 0.999
    gae_lambda: float = 0.97
    max_grad_norm: float = 0.5
    max_steps_per_episode: int = 2000
    reward_scale: float = 256.0
    completion_bonus: float = 5.0
    unsuccessful_penalty: float = 256.0
    device: str = 'cpu'

    def __post_init__(self):
        for field in ('num_envs','rollout_steps','minibatch_size','epochs','hidden_dim','max_steps_per_episode'):
            if type(getattr(self,field)) is not int or getattr(self,field)<=0:
                raise ValueError(field+' must be a positive integer')
        if self.learning_rate<=0 or self.reward_scale<=0 or not 0<self.clip_range<1:
            raise ValueError('invalid PPO learning rate, reward scale or clip range')
        if not 0<=self.gamma<=1 or not 0<=self.gae_lambda<=1:
            raise ValueError('invalid return discount')
        if self.entropy_coefficient<0 or self.value_coefficient<0 or self.unsuccessful_penalty<0:
            raise ValueError('negative PPO loss/reward coefficient')


def generalized_advantages(rewards, values, dones, last_values, *, gamma, gae_lambda):
    """Bootstrap only within an episode, including simultaneous vector resets."""
    rewards=np.asarray(rewards,dtype=np.float32)
    values=np.asarray(values,dtype=np.float32)
    dones=np.asarray(dones,dtype=np.bool_)
    if rewards.shape!=values.shape or rewards.shape!=dones.shape or rewards.ndim!=2:
        raise ValueError('rollouts must have matching time by environment arrays')
    following=np.asarray(last_values,dtype=np.float32)
    if following.shape!=rewards.shape[1:]:
        raise ValueError('bootstrap values do not match environments')
    advantages=np.zeros_like(rewards)
    carried=np.zeros_like(following)
    for step in reversed(range(len(rewards))):
        live=(~dones[step]).astype(np.float32)
        delta=rewards[step]+gamma*following*live-values[step]
        carried=delta+gamma*gae_lambda*live*carried
        advantages[step]=carried
        following=values[step]
    return advantages,advantages+values


class PPOTrainer:
    CHECKPOINT_FORMAT=1

    def __init__(self,simulator,config:PPOConfig|None=None,*,model=None):
        torch=_torch()
        from .network import GraphPolicyValueNetwork,NetworkConfig
        self.simulator=simulator
        if simulator.simulation_profile!='synthetic':
            raise ValueError('automatic PPO rollouts require the explicitly synthetic simulator profile')
        self.config=config or PPOConfig()
        self.policy_constraints=AUTONOMOUS_POLICY_CONSTRAINTS
        self.environment=PolicyConstrainedEnvironment(simulator,constraints=self.policy_constraints)
        self.encoder=FeatureEncoder(simulator,full_observability=False,constraints=self.policy_constraints)
        torch.manual_seed(self.config.seed)
        self.rng=random.Random(self.config.seed)
        self.numpy_rng=np.random.default_rng(self.config.seed)
        if model is None:
            model=GraphPolicyValueNetwork(NetworkConfig(
                node_feature_dim=self.encoder.node_feature_dim,
                global_feature_dim=self.encoder.global_feature_dim,
                option_feature_dim=self.encoder.option_feature_dim,
                hidden_dim=self.config.hidden_dim))
        self.model=model.to(self.config.device)
        if getattr(self.model.config,'dropout',0)!=0:
            raise ValueError('PPO requires dropout=0 so rollout and update policy probabilities agree')
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.config.learning_rate,weight_decay=1e-5)
        self.training_seeds:set[int]=set()
        self.next_seed_index=0
        self.states:list[GameState]=[]
        self.episode_seeds:list[int]=[]
        self.episode_returns:list[float]=[]
        self.updates_completed=0
        self.episodes_completed=0
        self.samples_seen=0
        self.learning_history=[]

    @staticmethod
    def _reserved_seed(seed):
        return 900000<=seed<900064 or 910000<=seed<910128

    def _reset_environment(self,index):
        seed=self.config.training_seed_start+self.next_seed_index
        self.next_seed_index+=1
        while seed in self.training_seeds:
            seed=self.config.training_seed_start+self.next_seed_index
            self.next_seed_index+=1
        if self._reserved_seed(seed):
            raise ValueError('training must not consume development or holdout seeds')
        self.training_seeds.add(seed)
        state=self.simulator.reset(seed)
        if index==len(self.states):
            self.states.append(state)
            self.episode_seeds.append(seed)
            self.episode_returns.append(0.0)
        else:
            self.states[index]=state
            self.episode_seeds[index]=seed
            self.episode_returns[index]=0.0

    def _tensors(self,encoded):
        torch=_torch()
        arrays=stack_encoded(encoded)
        return {name:torch.as_tensor(value,device=self.config.device) for name,value in arrays.items()}

    def choose_action(self,state,*,deterministic=False,rng=None):
        """Every action is selected by the network from the scope-legal menu."""
        torch=_torch()
        encoded=self.encoder.encode(state)
        legal=np.flatnonzero(encoded.action_mask)
        if not len(legal):
            raise RuntimeError('no scope-legal action for neural control')
        self.model.eval()
        with torch.inference_mode():
            logits,_=self.model(**self._tensors([encoded]))
            scores=logits[0,torch.as_tensor(legal,device=self.config.device)]
            if deterministic:
                return int(legal[int(scores.argmax().item())])
            probabilities=torch.softmax(scores,dim=0).cpu().numpy()
        chooser=rng or self.rng
        return int(chooser.choices(legal.tolist(),weights=probabilities.tolist(),k=1)[0])

    def _reward(self,before,after,*,unsuccessful=False):
        old=sum(item.category=='RELIC' for item in before.item_instances)
        new=sum(item.category=='RELIC' for item in after.item_instances)
        amount=float(new-old)
        if unsuccessful:
            amount-=self.config.unsuccessful_penalty
        elif after.terminal:
            amount+=self.config.completion_bonus
        # Money, preferred nodes, withdrawals, corn counts, gear and chases
        # receive no hand-strategy reward. Their worth must be learned from
        # resulting true relics and completion of the requested ending.
        return amount/self.config.reward_scale

    def collect_rollout(self):
        torch=_torch()
        while len(self.states)<self.config.num_envs:
            self._reset_environment(len(self.states))
        encoded_rows=[]
        action_rows=[]
        log_prob_rows=[]
        value_rows=[]
        reward_rows=[]
        done_rows=[]
        learning_rows=[]
        reports=[]
        coverage_examples=[]
        self.model.eval()
        for _ in range(self.config.rollout_steps):
            encoded=[self.encoder.encode(state) for state in self.states]
            if any(not item.action_mask.any() for item in encoded):
                raise RuntimeError('nonterminal rollout state has no scope-legal action')
            with torch.inference_mode():
                logits,values=self.model(**self._tensors(encoded))
                distribution=torch.distributions.Categorical(logits=logits)
                selected=distribution.sample()
                old_log_probs=distribution.log_prob(selected)
            actions=selected.cpu().numpy()
            predicted_values=values.cpu().numpy()
            rewards=np.zeros(self.config.num_envs,dtype=np.float32)
            dones=np.zeros(self.config.num_envs,dtype=np.bool_)
            learning=np.ones(self.config.num_envs,dtype=np.bool_)
            for index,action_id in enumerate(actions):
                before=self.states[index]
                error=None
                coverage_blocked=None
                scope_deadend=False
                try:
                    after=self.environment.transition(before,int(action_id)).next_state
                except Exception as exc:
                    after=before
                    error=f'{type(exc).__name__}: {exc}'
                    coverage_blocked='simulator_transition_error'
                if not after.terminal and error is None:
                    if not self.simulator.legal_actions(after):
                        node=after.floor_map.node(after.pending_node_id or after.current_node_id)
                        coverage_blocked=('awaiting_external_observation' if node.requires_observation
                            else 'nonterminal_without_game_actions')
                    else:
                        try:
                            scope_deadend=not self.environment.legal_action_ids(after)
                        except RuntimeError:
                            scope_deadend=True
                truncated=not after.terminal and after.step_count>=self.config.max_steps_per_episode
                violations=scope_violations(after,self.policy_constraints) if after.terminal else []
                if scope_deadend:
                    violations.append('no action remains within the ending-one task scope')
                unsuccessful=bool(truncated or violations)
                actual_reward=self._reward(before,after,unsuccessful=unsuccessful)
                rewards[index]=actual_reward
                self.episode_returns[index]+=float(actual_reward)
                if coverage_blocked:
                    # An unimplemented branch is missing data, not evidence
                    # that the corresponding real game action is bad. Censor
                    # that transition's losses and bootstrap the preceding
                    # trajectory from this pre-action value; do not fabricate
                    # a decline, replacement reward, or failure penalty.
                    learning[index]=False
                    rewards[index]=predicted_values[index]
                    coverage_examples.append({'seed':self.episode_seeds[index],
                        'action_id':int(action_id),'before':before,'after':after,
                        'reason':coverage_blocked,'error':error})
                dones[index]=after.terminal or unsuccessful or bool(coverage_blocked)
                self.states[index]=after
                if dones[index]:
                    failure_node=after.floor_map.node(after.pending_node_id or after.current_node_id)
                    reports.append({'seed':self.episode_seeds[index],'steps':after.step_count,
                        'completed':bool(after.terminal and not unsuccessful and not coverage_blocked),'ending_id':after.ending_id,
                        'final_relics':sum(item.category=='RELIC' for item in after.item_instances),
                        'acquired_relics':sum(entry.quantity for entry in after.ledger
                            if entry.operation=='acquire' and entry.category=='RELIC'),
                        'normalized_return':self.episode_returns[index],
                        'truncated':truncated,'scope_violations':violations,'error':error,
                        'coverage_blocked':coverage_blocked,
                        'last_node_id':failure_node.node_id,'last_node_type':failure_node.node_type.value,
                        'last_event_name':failure_node.event_name,'last_action_id':int(action_id)})
                    self.episodes_completed+=1
                    self._reset_environment(index)
            encoded_rows.extend(encoded)
            action_rows.append(actions.copy())
            log_prob_rows.append(old_log_probs.cpu().numpy())
            value_rows.append(predicted_values)
            reward_rows.append(rewards)
            done_rows.append(dones)
            learning_rows.append(learning)
        with torch.inference_mode():
            _,last_values=self.model(**self._tensors([self.encoder.encode(state) for state in self.states]))
        values=np.asarray(value_rows,dtype=np.float32)
        advantages,returns=generalized_advantages(reward_rows,values,done_rows,last_values.cpu().numpy(),
            gamma=self.config.gamma,gae_lambda=self.config.gae_lambda)
        self.samples_seen+=advantages.size
        return {'encoded':encoded_rows,'actions':np.asarray(action_rows).reshape(-1),
            'log_probs':np.asarray(log_prob_rows).reshape(-1),'values':values.reshape(-1),
            'advantages':advantages.reshape(-1),'returns':returns.reshape(-1),
            'learning_mask':np.asarray(learning_rows).reshape(-1),'episodes':reports,
            'coverage_examples':coverage_examples}

    def update(self,rollout):
        torch=_torch()
        count=len(rollout['actions'])
        eligible=np.flatnonzero(rollout.get('learning_mask',np.ones(count,dtype=np.bool_)))
        advantages=rollout['advantages']
        if len(eligible):
            advantages=(advantages-advantages[eligible].mean())/(advantages[eligible].std()+1e-8)
        losses=[]
        self.model.train()
        for _ in range(self.config.epochs):
            order=self.numpy_rng.permutation(eligible)
            for start in range(0,len(eligible),self.config.minibatch_size):
                selected=order[start:start+self.config.minibatch_size]
                batch=self._tensors([rollout['encoded'][int(index)] for index in selected])
                logits,values=self.model(**batch)
                distribution=torch.distributions.Categorical(logits=logits)
                actions=torch.as_tensor(rollout['actions'][selected],device=self.config.device,dtype=torch.long)
                log_probs=distribution.log_prob(actions)
                old_log=torch.as_tensor(rollout['log_probs'][selected],device=self.config.device)
                advantage=torch.as_tensor(advantages[selected],device=self.config.device)
                target=torch.as_tensor(rollout['returns'][selected],device=self.config.device)
                ratio=(log_probs-old_log).exp()
                clipped=ratio.clamp(1-self.config.clip_range,1+self.config.clip_range)
                policy_loss=-torch.minimum(ratio*advantage,clipped*advantage).mean()
                value_loss=torch.nn.functional.mse_loss(values,target)
                entropy=distribution.entropy().mean()
                loss=policy_loss+self.config.value_coefficient*value_loss-self.config.entropy_coefficient*entropy
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl=((ratio-1)-(log_probs-old_log)).mean()
                    clip_fraction=((ratio-1).abs()>self.config.clip_range).float().mean()
                losses.append([float(value.detach().cpu()) for value in
                    (loss,policy_loss,value_loss,entropy,approx_kl,clip_fraction)])
        self.updates_completed+=1
        result={key:float(value) for key,value in zip(
            ('loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction'),
            np.mean(losses,axis=0) if losses else np.zeros(6))}
        result.update(update=self.updates_completed,episodes_completed=self.episodes_completed,
            samples_seen=self.samples_seen,learning_samples=len(eligible),
            censored_transitions=count-len(eligible),episodes=rollout['episodes'])
        return result

    @property
    def implementation_sha256(self):
        return sha256(Path(__file__).read_bytes()).hexdigest()

    @property
    def network_implementation_sha256(self):
        return sha256(Path(__file__).with_name('network.py').read_bytes()).hexdigest()

    def save_checkpoint(self,path):
        torch=_torch()
        path=Path(path).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={'format_version':self.CHECKPOINT_FORMAT,'algorithm':'PPO_DIRECT_NEURAL',
            'environment_sha256':self.simulator.environment_sha256,
            'map_generator_config':asdict(self.simulator.map_generator.config),
            'economy_config':asdict(self.simulator.economy.config),
            'policy_constraints':asdict(self.policy_constraints),
            'feature_schema':self.encoder.schema,'feature_schema_sha256':self.encoder.schema_sha256,
            'implementation_sha256':self.implementation_sha256,
            'network_implementation_sha256':self.network_implementation_sha256,
            'training_config':asdict(self.config),'network_config':asdict(self.model.config),
            'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),
            'training_seeds':sorted(self.training_seeds),'next_seed_index':self.next_seed_index,
            'updates_completed':self.updates_completed,'episodes_completed':self.episodes_completed,
            'samples_seen':self.samples_seen,'python_random_state':self.rng.getstate(),
            'numpy_random_state':self.numpy_rng.bit_generator.state,'torch_random_state':torch.random.get_rng_state(),
            'states':self.states,'episode_seeds':self.episode_seeds,'episode_returns':self.episode_returns,
            'learning_history':self.learning_history}
        temporary=path.with_name(path.name+'.tmp')
        torch.save(payload,temporary)
        os.replace(temporary,path)
        return path

    @classmethod
    def warm_start(cls, simulator, path, config=None):
        """Explicit weights-only migration; no stale worlds, optimizer or returns.

        Observation/action and network semantics must still match exactly.
        Source seeds and hashes remain in the new training lineage.
        """
        torch=_torch()
        from .network import GraphPolicyValueNetwork, NetworkConfig
        path=Path(path)
        settings=config or PPOConfig()
        source=torch.load(path,map_location=settings.device,weights_only=False)
        if (source.get('format_version')!=cls.CHECKPOINT_FORMAT
                or source.get('algorithm')!='PPO_DIRECT_NEURAL'):
            raise ValueError('warm start requires a direct-neural PPO checkpoint')
        model=GraphPolicyValueNetwork(NetworkConfig(**source['network_config']))
        if settings.hidden_dim != model.config.hidden_dim:
            raise ValueError('warm start hidden dimension must match the source network')
        result=cls(simulator,settings,model=model)
        if (source.get('feature_schema_sha256')!=result.encoder.schema_sha256
                or source.get('policy_constraints')!=asdict(result.policy_constraints)
                or source.get('network_implementation_sha256')!=result.network_implementation_sha256):
            raise ValueError('warm start observation, action or network semantics differ')
        result.training_seeds=set(source['training_seeds'])
        if any(result._reserved_seed(seed) for seed in result.training_seeds):
            raise ValueError('warm start history overlaps reserved evaluation seeds')
        model.load_state_dict(source['model_state_dict'],strict=True)
        result.learning_history=list(source.get('learning_history',[]))+[{
            'type':'explicit_weights_only_environment_migration',
            'source_checkpoint':str(path.resolve()),
            'source_checkpoint_sha256':sha256(path.read_bytes()).hexdigest(),
            'source_environment_sha256':source['environment_sha256'],
            'destination_environment_sha256':simulator.environment_sha256,
            'source_learner_sha256':source['implementation_sha256'],
            'source_training_seed_count':len(result.training_seeds),
            'optimizer_and_rollouts_reset':True}]
        return result

    @classmethod
    def load_checkpoint(cls,simulator,path,*,device='cpu'):
        torch=_torch()
        from .network import GraphPolicyValueNetwork,NetworkConfig
        payload=torch.load(Path(path),map_location=device,weights_only=False)
        if payload.get('format_version')!=cls.CHECKPOINT_FORMAT or payload.get('algorithm')!='PPO_DIRECT_NEURAL':
            raise ValueError('checkpoint is not a supported direct-neural PPO model')
        if payload.get('environment_sha256')!=simulator.environment_sha256:
            raise ValueError('PPO checkpoint environment semantics differ')
        if payload['network_config'].get('architecture_version')!=2:
            raise ValueError('PPO checkpoint network architecture differs')
        settings={**payload['training_config'],'device':device}
        model=GraphPolicyValueNetwork(NetworkConfig(**payload['network_config']))
        result=cls(simulator,PPOConfig(**settings),model=model)
        if (payload.get('feature_schema_sha256')!=result.encoder.schema_sha256
                or payload.get('policy_constraints')!=asdict(result.policy_constraints)):
            raise ValueError('PPO checkpoint observation or action scope differs')
        if payload.get('implementation_sha256')!=result.implementation_sha256:
            raise ValueError('PPO checkpoint learner semantics differ')
        if payload.get('network_implementation_sha256')!=result.network_implementation_sha256:
            raise ValueError('PPO checkpoint network implementation differs')
        model.load_state_dict(payload['model_state_dict'])
        result.optimizer.load_state_dict(payload['optimizer_state_dict'])
        result.training_seeds=set(payload['training_seeds'])
        result.learning_history=payload.get('learning_history',[])
        if any(result._reserved_seed(seed) for seed in result.training_seeds):
            raise ValueError('PPO training history overlaps reserved evaluation seeds')
        for field in ('next_seed_index','updates_completed','episodes_completed','samples_seen',
                'states','episode_seeds','episode_returns'):
            setattr(result,field,payload[field])
        result.rng.setstate(payload['python_random_state'])
        result.numpy_rng.bit_generator.state=payload['numpy_random_state']
        torch.random.set_rng_state(payload['torch_random_state'])
        return result
