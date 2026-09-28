"""Public equipment execution commitment around the unchanged neural policy.

An equipment UI change reveals no reward. If the public progress context is
unchanged and a legal move exists, let the network execute a destination before
changing equipment again. This is a decoder probe, not a trained network.
"""
from hashlib import sha256
from pathlib import Path
import torch

from .neural_memory_v2 import FairEquipmentCycleMemory, public_progress_key


class FairEquipmentCommitMemory(FairEquipmentCycleMemory):
    VERSION = 'PUBLIC_EQUIPMENT_COMMIT_V2_FAIR'

    def begin_episode(self):
        super().begin_episode()
        self.commit_progress = None
        self.committed_decisions = 0

    @property
    def implementation_sha256(self):
        return sha256(Path(__file__).read_bytes() + super().implementation_sha256.encode()).hexdigest()

    def choose_action(self, state, *, deterministic=False, rng=None):
        progress = public_progress_key(state)
        allowed = self.encoder.encode(state).action_mask
        actions = [a for a in self.simulator.legal_actions(state) if allowed[a.action_id]]
        equipment = [a.action_id for a in actions if a.kind.value == 'EQUIP']
        block = self.commit_progress == progress and any(a.kind.value == 'MOVE' for a in actions)
        hook = None
        if block and equipment:
            def suppress(module, inputs, outputs):
                logits, values = outputs
                logits = logits.clone()
                logits[:, equipment] = torch.finfo(logits.dtype).min
                return logits, values
            hook = self.model.register_forward_hook(suppress)
            self.committed_decisions += 1
        try:
            chosen = super().choose_action(state, deterministic=deterministic, rng=rng)
        finally:
            if hook is not None:
                hook.remove()
        self.commit_progress = progress if chosen in equipment else None
        return chosen
