"""PPO with telescoping public-liquidity potential, leaving game returns intact."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .ppo import PPOConfig,PPOTrainer,_torch


class PotentialPPOTrainer(PPOTrainer):
    SHAPING='discounted_public_liquidity_potential_v1'

    def potential(self,state):
        if state.terminal:
            return 0.0
        liquidity=state.resources.gold+sum(self.simulator.economy.quote_sell(state,item)
            for item in state.item_instances if item.category in ('MOVE','GOODS','PASSIVE'))
        return min(max(0,liquidity)/8.0,128.0)/self.config.reward_scale

    def _reward(self,before,after,*,unsuccessful=False):
        base=super()._reward(before,after,unsuccessful=unsuccessful)
        following=0.0 if unsuccessful else self.potential(after)
        return base+self.config.gamma*following-self.potential(before)

    @property
    def implementation_sha256(self):
        return sha256(Path(__file__).read_bytes()+Path(__file__).with_name('ppo.py').read_bytes()).hexdigest()

    def save_checkpoint(self,path):
        path=super().save_checkpoint(path)
        torch=_torch()
        data=torch.load(path,map_location='cpu',weights_only=False)
        data['reward_shaping']=self.SHAPING
        data['potential_definition']='min(max(0, observed gold + actual part sale quotes)/8, 128)/reward_scale; terminal potential zero'
        temporary=path.with_name(path.name+'.tmp')
        torch.save(data,temporary)
        temporary.replace(path)
        return path

    @classmethod
    def load_checkpoint(cls,simulator,path,*,device='cpu'):
        torch=_torch()
        from .network import GraphPolicyValueNetwork,NetworkConfig
        data=torch.load(path,map_location=device,weights_only=False)
        if data.get('reward_shaping')!=cls.SHAPING or data.get('format_version')!=cls.CHECKPOINT_FORMAT:
            raise ValueError('unsupported potential checkpoint')
        model=GraphPolicyValueNetwork(NetworkConfig(**data['network_config']))
        result=cls(simulator,PPOConfig(**{**data['training_config'],'device':device}),model=model)
        checks={'environment_sha256':simulator.environment_sha256,
            'feature_schema_sha256':result.encoder.schema_sha256,
            'implementation_sha256':result.implementation_sha256,
            'network_implementation_sha256':result.network_implementation_sha256,
            'policy_constraints':asdict(result.policy_constraints)}
        if any(data.get(key)!=value for key,value in checks.items()):
            raise ValueError('potential checkpoint semantics differ')
        result.model.load_state_dict(data['model_state_dict'])
        result.optimizer.load_state_dict(data['optimizer_state_dict'])
        result.training_seeds=set(data['training_seeds'])
        if any(result._reserved_seed(seed) for seed in result.training_seeds):
            raise ValueError('potential checkpoint training overlaps evaluation')
        for name in ('next_seed_index','updates_completed','episodes_completed','samples_seen',
                     'states','episode_seeds','episode_returns'):
            setattr(result,name,data[name])
        result.learning_history=data.get('learning_history',[])
        result.rng.setstate(data['python_random_state'])
        result.numpy_rng.bit_generator.state=data['numpy_random_state']
        torch.random.set_rng_state(data['torch_random_state'])
        return result
