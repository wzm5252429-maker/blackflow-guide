"""Observed-state memory for provably unproductive equipment-switch cycles."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch


class EquipmentCycleMemory:
    def __init__(self,trainer):
        self.trainer=trainer
        self.begin_episode()

    def __getattr__(self,name):
        return getattr(self.trainer,name)

    @property
    def implementation_sha256(self):
        return sha256(Path(__file__).read_bytes()+self.trainer.implementation_sha256.encode()).hexdigest()

    def begin_episode(self):
        self.seen={}
        self.prevented_repeats=0

    def choose_action(self,state,*,deterministic=False,rng=None):
        encoded=self.encoder.encode(state)
        digest=sha256()
        for field in encoded.__dataclass_fields__:
            digest.update(getattr(encoded,field).tobytes())
        key=digest.digest()
        prior=self.seen.get(key,set())
        legal=np.flatnonzero(encoded.action_mask).tolist()
        if not legal:
            raise RuntimeError('no scope-legal action for neural control')
        equipment={action for action in legal if self.simulator.decode_action(state,action).kind.value=='EQUIP'}
        candidates=[action for action in legal if action not in prior or action not in equipment]
        self.prevented_repeats+=len(legal)-len(candidates)
        if not candidates:
            # Preserve genuine game legality in a menu with no alternative.
            candidates=legal
        self.model.eval()
        with torch.inference_mode():
            logits,_=self.model(**self.trainer._tensors([encoded]))
            scores=logits[0,torch.as_tensor(candidates,device=self.config.device)]
            if deterministic:
                action=candidates[int(scores.argmax())]
            else:
                probabilities=torch.softmax(scores,dim=0).cpu().numpy().tolist()
                action=(rng or self.trainer.rng).choices(candidates,weights=probabilities,k=1)[0]
        if action in equipment:
            self.seen.setdefault(key,set()).add(action)
        else:
            # Only consecutive equipment switches are deduplicated. Any
            # actual movement, purchase, recruitment or other game action
            # starts a new context, preserving productive economic cycles.
            self.seen.clear()
        return int(action)
