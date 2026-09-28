"""Execute a neural equipment choice before reconsidering unchanged equipment.

This optional action-memory controller selects no economic goal or destination.
Newly revealed information or changed public resources reopen equipment choices.
"""
from hashlib import sha256
from pathlib import Path
import torch
from .neural_memory import EquipmentCycleMemory


class EquipmentCommitMemory(EquipmentCycleMemory):
    def begin_episode(self):
        super().begin_episode()
        self.commit_context=None
        self.committed_equipment_options_suppressed=0

    @property
    def implementation_sha256(self):
        return sha256(Path(__file__).read_bytes()+super().implementation_sha256.encode()).hexdigest()

    @staticmethod
    def public_context(state):
        return (state.floor,state.current_node_id,state.pending_node_id,state.resources,
            state.item_instances,state.revealed)

    def choose_action(self,state,*,deterministic=False,rng=None):
        context=self.public_context(state)
        legal=self.simulator.legal_actions(state)
        equipment=[action.action_id for action in legal if action.kind.value=='EQUIP']
        block=self.commit_context==context and any(action.kind.value=='MOVE' for action in legal)
        hook=None
        if block and equipment:
            def suppress(module,inputs,outputs):
                logits,values=outputs
                logits=logits.clone()
                logits[:,equipment]=torch.finfo(logits.dtype).min
                return logits,values
            hook=self.model.register_forward_hook(suppress)
            self.committed_equipment_options_suppressed+=len(equipment)
        try:
            action=super().choose_action(state,deterministic=deterministic,rng=rng)
        finally:
            if hook is not None:
                hook.remove()
        self.commit_context=context if action in equipment else None
        return action
