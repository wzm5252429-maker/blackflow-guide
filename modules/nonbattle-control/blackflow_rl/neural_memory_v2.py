"""Fair, public-identity-aware memory for zero-progress equipment cycles.

No game action is invented. Repeated equipment choices are suppressed until all
are tried. If a menu then contains only equipment actions, its least-tried set is
offered to the same neural scores. This makes navigation through a finite graph
of equipment states fair instead of indefinitely restoring its highest logit.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch


def public_progress_key(state):
    """Exact public resource/usage guard, independent of equipped-instance UI."""
    return (state.floor, state.current_node_id, state.pending_node_id,
            state.resources, state.item_instances, len(state.ledger),
            state.completed, state.revealed, state.bank_balance,
            state.supply_vouchers, state.parts_capacity, state.squad_capacity)


def semantic_equipment_action(action):
    return (action.kind.value, action.equipment_instance_id)


class FairEquipmentCycleMemory:
    VERSION = 'PUBLIC_EQUIPMENT_MEMORY_V2_FAIR_SEMANTIC_IDENTITY'

    def __init__(self, trainer):
        self.trainer = trainer
        self.begin_episode()

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    @property
    def implementation_sha256(self):
        return sha256(Path(__file__).read_bytes()+self.trainer.implementation_sha256.encode()).hexdigest()

    def begin_episode(self):
        self.seen = {}
        self.attempt_counts = {}
        self.last_progress = None
        self.prevented_repeats = 0
        self.fair_equipment_fallbacks = 0

    def _clear_context(self):
        self.seen.clear()
        self.attempt_counts.clear()
        self.last_progress = None

    def choose_action(self, state, *, deterministic=False, rng=None):
        progress = public_progress_key(state)
        if self.last_progress is not None and progress != self.last_progress:
            self._clear_context()
        encoded = self.encoder.encode(state)
        digest = sha256()
        for field in encoded.__dataclass_fields__:
            digest.update(getattr(encoded, field).tobytes())
        # Equal item type and remaining uses do not imply equal public identity.
        digest.update(repr(state.equipped_instance_id).encode())
        key = digest.digest()
        legal = np.flatnonzero(encoded.action_mask).tolist()
        if not legal:
            raise RuntimeError('no scope-legal action for neural control')
        decoded = {action:self.simulator.decode_action(state, action) for action in legal}
        equipment = {action:semantic_equipment_action(value) for action,value in decoded.items()
                     if value.kind.value == 'EQUIP'}
        prior = self.seen.get(key, set())
        candidates = [action for action in legal
                      if action not in equipment or equipment[action] not in prior]
        self.prevented_repeats += len(legal)-len(candidates)
        if not candidates:
            # By construction every legal choice here is EQUIP. Never select an
            # unseen transition or assume that an equipment item can reach a node.
            counts = self.attempt_counts.setdefault(key, Counter())
            minimum = min(counts[equipment[action]] for action in legal)
            candidates = [action for action in legal if counts[equipment[action]] == minimum]
            self.fair_equipment_fallbacks += 1
        self.model.eval()
        with torch.inference_mode():
            logits,_ = self.model(**self.trainer._tensors([encoded]))
            scores = logits[0,torch.as_tensor(candidates, device=self.config.device)]
            if deterministic:
                action = candidates[int(scores.argmax())]
            else:
                probabilities = scores.softmax(0).cpu().tolist()
                action = (rng or self.trainer.rng).choices(candidates, weights=probabilities, k=1)[0]
        if action in equipment:
            semantic = equipment[action]
            self.seen.setdefault(key, set()).add(semantic)
            self.attempt_counts.setdefault(key, Counter())[semantic] += 1
            self.last_progress = progress
        else:
            # Movement, buying, selling, recruitment and all other real actions
            # start a fresh context. Changed exact resources/uses also reset it.
            self._clear_context()
        return int(action)
