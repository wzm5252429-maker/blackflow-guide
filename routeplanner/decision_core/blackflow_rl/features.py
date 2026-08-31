from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .domain import GameState, NodeType
from .simulator import BlackflowSimulator


OBSERVED_NODE_LABELS = tuple(item.value for item in NodeType) + (
    "UNKNOWN_MYSTERY",
    "UNKNOWN_FEROCITY",
)
_NODE_LABEL_INDEX = {label: index for index, label in enumerate(OBSERVED_NODE_LABELS)}

RESOURCE_FIELDS = (
    "hp",
    "max_hp",
    "shield",
    "gold",
    "hope",
    "parts",
    "relics",
    "tickets",
    "team_strength",
    "action_points",
)
RESOURCE_SCALES = np.asarray((8, 8, 10, 20, 10, 5, 5, 5, 10, 5), dtype=np.float32)
INVENTORY_FLAGS = ("alpha", "beta", "beacon", "cage", "ending_3_key", "processed")


@dataclass(frozen=True, slots=True)
class EncodedState:
    node_features: np.ndarray
    adjacency: np.ndarray
    node_mask: np.ndarray
    global_features: np.ndarray
    option_features: np.ndarray
    action_mask: np.ndarray

    def as_torch(self, device: str | None = None) -> dict[str, Any]:
        """Convert one sample without importing torch at package import time."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on minimal installs
            raise RuntimeError(
                "PyTorch is required for neural inference; install the training dependencies"
            ) from exc
        target = torch.device(device) if device is not None else None
        return {
            "node_features": torch.as_tensor(self.node_features, device=target),
            "adjacency": torch.as_tensor(self.adjacency, device=target),
            "node_mask": torch.as_tensor(self.node_mask, device=target),
            "global_features": torch.as_tensor(self.global_features, device=target),
            "option_features": torch.as_tensor(self.option_features, device=target),
            "action_mask": torch.as_tensor(self.action_mask, device=target),
        }


class FeatureEncoder:
    """Permutation-equivariant padded graph and candidate-action encoder."""

    NODE_SCALAR_DIM = 12
    GLOBAL_BASE_DIM = 14
    OPTION_EXTRA_DIM = 4

    def __init__(
        self,
        simulator: BlackflowSimulator,
        *,
        full_observability: bool = False,
    ) -> None:
        self.simulator = simulator
        self.ruleset = simulator.ruleset
        self.full_observability = full_observability

    @property
    def node_feature_dim(self) -> int:
        return len(OBSERVED_NODE_LABELS) + self.NODE_SCALAR_DIM

    @property
    def global_feature_dim(self) -> int:
        return self.GLOBAL_BASE_DIM + len(INVENTORY_FLAGS)

    @property
    def option_feature_dim(self) -> int:
        return len(RESOURCE_FIELDS) + self.OPTION_EXTRA_DIM

    def encode(self, state: GameState) -> EncodedState:
        max_nodes = self.ruleset.max_nodes
        max_options = self.ruleset.max_options
        floor_map = state.floor_map
        node_features = np.zeros(
            (max_nodes, self.node_feature_dim), dtype=np.float32
        )
        adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        node_mask = np.zeros(max_nodes, dtype=np.bool_)
        option_features = np.zeros(
            (max_options, self.option_feature_dim), dtype=np.float32
        )
        action_mask = np.zeros(max_nodes + max_options, dtype=np.bool_)

        frontier = self.simulator.reachable_frontier(state) if not state.terminal else {}
        base_ap = self.ruleset.floor(state.floor).action_points
        for node in floor_map.nodes:
            node_mask[node.index] = True
            observed = self.full_observability or node.node_id in state.revealed
            if observed:
                type_label = node.node_type.value
            else:
                type_label = (
                    "UNKNOWN_FEROCITY" if node.is_battle else "UNKNOWN_MYSTERY"
                )
            node_features[node.index, _NODE_LABEL_INDEX[type_label]] = 1.0
            offset = len(OBSERVED_NODE_LABELS)
            movement_cost = frontier.get(node.node_id, 0)
            node_features[node.index, offset:] = np.asarray(
                (
                    node.node_id == state.current_node_id,
                    node.node_id in state.completed,
                    observed,
                    node.is_exit,
                    node.node_id == state.pending_node_id,
                    node.node_id in frontier,
                    node.row / max(1, floor_map.height - 1),
                    node.col / max(1, floor_map.width - 1),
                    node.distance_from_start / 15.0,
                    movement_cost / max(1, base_ap),
                    bool(node.options) and node.node_id == state.pending_node_id,
                    node.is_battle,
                ),
                dtype=np.float32,
            )

        id_to_index = {node.node_id: node.index for node in floor_map.nodes}
        for left, right in floor_map.edges:
            left_index, right_index = id_to_index[left], id_to_index[right]
            adjacency[left_index, right_index] = 1.0
            adjacency[right_index, left_index] = 1.0

        legal_ids = self.simulator.legal_action_ids(state)
        for action_id in legal_ids:
            action_mask[action_id] = True

        if state.pending_node_id is not None:
            pending = floor_map.node(state.pending_node_id)
            for index, option in enumerate(pending.options[:max_options]):
                delta = np.asarray(
                    [getattr(option.effect, name) for name in RESOURCE_FIELDS],
                    dtype=np.float32,
                ) / RESOURCE_SCALES
                option_features[index] = np.concatenate(
                    (
                        delta,
                        np.asarray(
                            (
                                option.battle,
                                len(option.add_items) / 3.0,
                                len(option.remove_items) / 3.0,
                                option.is_available(state.resources, state.inventory),
                            ),
                            dtype=np.float32,
                        ),
                    )
                )

        resources = state.resources
        completed_here = sum(
            node.node_id in state.completed for node in floor_map.nodes
        )
        global_base = np.asarray(
            (
                (state.floor - 1) / max(1, len(state.maps) - 1),
                completed_here / max(1, len(floor_map.nodes)),
                resources.action_points / max(1, base_ap),
                resources.hp / max(1, resources.max_hp),
                resources.max_hp / 20.0,
                resources.shield / 20.0,
                resources.gold / 50.0,
                resources.hope / 30.0,
                resources.parts / 10.0,
                resources.relics / 10.0,
                resources.tickets / 10.0,
                resources.team_strength / 20.0,
                state.pending_node_id is not None,
                state.chase_count / max(1, len(state.maps)),
            ),
            dtype=np.float32,
        )
        inventory = np.asarray(
            [flag in state.inventory for flag in INVENTORY_FLAGS], dtype=np.float32
        )
        global_features = np.concatenate((global_base, inventory))

        return EncodedState(
            node_features=node_features,
            adjacency=adjacency,
            node_mask=node_mask,
            global_features=global_features,
            option_features=option_features,
            action_mask=action_mask,
        )


def stack_encoded(samples: list[EncodedState]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot stack an empty encoded batch")
    return {
        field: np.stack([getattr(sample, field) for sample in samples], axis=0)
        for field in EncodedState.__dataclass_fields__
    }
