"""Batched graph policy/value network for Black Flow Tree Sea training.

The module intentionally depends only on PyTorch.  It accepts padded batches of
graphs and event options, produces one policy logit per node/option action, and
predicts a scalar value in ``[-1, 1]`` for every batch item.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PathType: TypeAlias = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Architecture and input dimensions for :class:`GraphPolicyValueNetwork`.

    Attributes:
        node_feature_dim: Number of features attached to each graph node.
        global_feature_dim: Number of map/player-level input features.
        option_feature_dim: Number of features attached to each event option.
        hidden_dim: Width used by encoders, message passing, and output heads.
        num_message_passing_layers: Number of graph layers; restricted to 2--3
            to keep inference inexpensive while covering the requested receptive
            field.
        dropout: Dropout probability used after hidden activations.
        architecture_version: Explicit schema for menu-aware policy/value heads.
    """

    node_feature_dim: int
    global_feature_dim: int
    option_feature_dim: int
    hidden_dim: int = 128
    num_message_passing_layers: int = 3
    dropout: float = 0.0
    architecture_version: int = 2

    def __post_init__(self) -> None:
        """Reject invalid dimensions early, including when loading checkpoints."""

        dimension_fields = (
            "node_feature_dim",
            "global_feature_dim",
            "option_feature_dim",
            "hidden_dim",
        )
        for field_name in dimension_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

        if self.num_message_passing_layers not in (2, 3):
            raise ValueError(
                "num_message_passing_layers must be either 2 or 3, "
                f"got {self.num_message_passing_layers!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")
        if type(self.architecture_version) is not int or self.architecture_version != 2:
            raise ValueError(
                "architecture_version must be 2 for the menu-aware network, "
                f"got {self.architecture_version!r}"
            )


class _DirectedMessagePassing(nn.Module):
    """Residual mean-aggregation layer for directed, padded graphs."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.outgoing_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.incoming_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        outgoing_adjacency: Tensor,
        incoming_adjacency: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Update node states using separately parameterized edge directions."""

        outgoing_messages = torch.bmm(outgoing_adjacency, hidden)
        incoming_messages = torch.bmm(incoming_adjacency, hidden)
        update = (
            self.self_projection(hidden)
            + self.outgoing_projection(outgoing_messages)
            + self.incoming_projection(incoming_messages)
            + self.bias
        )
        update = self.dropout(F.gelu(update))
        hidden = self.normalization(hidden + update)

        # torch.where (instead of multiplication) also suppresses NaNs that may
        # occur in data stored in padded positions.
        return torch.where(node_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))


class GraphPolicyValueNetwork(nn.Module):
    """A batched graph policy/value model with no graph-library dependency.

    ``adjacency[b, i, j]`` denotes a directed edge from node ``i`` to node
    ``j``.  Each graph layer aggregates both outgoing and incoming neighbours
    using separate projections.  Invalid/padded entries are excluded before
    message passing and pooling.

    The policy action order is all ``N`` node actions followed by all ``O``
    option actions.  ``option_observation_mask`` identifies displayed options,
    including unaffordable purchases, while ``action_mask`` controls selection.
    Mean/max menu pooling lets every action and the value head compare the
    displayed choices without depending on their order.
    """

    CHECKPOINT_FORMAT_VERSION = 2

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(config.global_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(config.option_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.menu_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.message_passing = nn.ModuleList(
            _DirectedMessagePassing(hidden_dim, config.dropout)
            for _ in range(config.num_message_passing_layers)
        )

        policy_input_dim = hidden_dim * 4
        self.node_policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.option_policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use stable defaults and small initial policy/value predictions."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for head in (self.node_policy_head, self.option_policy_head, self.value_head):
            output_layer = next(
                module for module in reversed(head) if isinstance(module, nn.Linear)
            )
            nn.init.uniform_(output_layer.weight, -1.0e-3, 1.0e-3)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    @property
    def device(self) -> torch.device:
        """Device on which the module's parameters currently reside."""

        return self.node_encoder[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype used by the module's parameters."""

        return self.node_encoder[0].weight.dtype

    def _validate_inputs(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        option_features: Tensor,
        action_mask: Tensor,
        option_observation_mask: Tensor | None = None,
    ) -> tuple[int, int, int]:
        """Validate public tensor ranks and mutually dependent dimensions."""

        expected_ranks = {
            "node_features": (node_features, 3),
            "adjacency": (adjacency, 3),
            "node_mask": (node_mask, 2),
            "global_features": (global_features, 2),
            "option_features": (option_features, 3),
            "action_mask": (action_mask, 2),
        }
        if option_observation_mask is not None:
            expected_ranks["option_observation_mask"] = (option_observation_mask, 2)
        for name, (tensor, rank) in expected_ranks.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim != rank:
                raise ValueError(
                    f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}"
                )

        batch_size, node_count, node_feature_dim = node_features.shape
        option_batch_size, option_count, option_feature_dim = option_features.shape

        if node_feature_dim != self.config.node_feature_dim:
            raise ValueError(
                "node_features has feature dimension "
                f"{node_feature_dim}, expected {self.config.node_feature_dim}"
            )
        if global_features.shape != (batch_size, self.config.global_feature_dim):
            raise ValueError(
                "global_features must have shape "
                f"({batch_size}, {self.config.global_feature_dim}), got "
                f"{tuple(global_features.shape)}"
            )
        if option_feature_dim != self.config.option_feature_dim:
            raise ValueError(
                "option_features has feature dimension "
                f"{option_feature_dim}, expected {self.config.option_feature_dim}"
            )
        if option_batch_size != batch_size:
            raise ValueError(
                f"option_features batch size {option_batch_size} does not match {batch_size}"
            )
        if adjacency.shape != (batch_size, node_count, node_count):
            raise ValueError(
                "adjacency must have shape "
                f"({batch_size}, {node_count}, {node_count}), got {tuple(adjacency.shape)}"
            )
        if node_mask.shape != (batch_size, node_count):
            raise ValueError(
                f"node_mask must have shape ({batch_size}, {node_count}), "
                f"got {tuple(node_mask.shape)}"
            )
        if action_mask.shape != (batch_size, node_count + option_count):
            raise ValueError(
                "action_mask must have shape "
                f"({batch_size}, {node_count + option_count}), got "
                f"{tuple(action_mask.shape)}"
            )
        if option_observation_mask is not None and option_observation_mask.shape != (
            batch_size, option_count
        ):
            raise ValueError(
                "option_observation_mask must have shape "
                f"({batch_size}, {option_count}), got "
                f"{tuple(option_observation_mask.shape)}"
            )
        return batch_size, node_count, option_count

    @staticmethod
    def _normalize_adjacency(adjacency: Tensor) -> Tensor:
        """Row-normalize weighted adjacency while leaving isolated rows at zero."""

        degree = adjacency.abs().sum(dim=-1, keepdim=True)
        safe_degree = degree.clamp_min(torch.finfo(adjacency.dtype).eps)
        return adjacency / safe_degree

    @staticmethod
    def _masked_mean(hidden: Tensor, node_mask: Tensor) -> Tensor:
        """Mean-pool valid nodes; an empty graph maps to an all-zero vector."""

        clean_hidden = torch.where(
            node_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden)
        )
        count = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return clean_hidden.sum(dim=1) / count

    @staticmethod
    def _masked_max(hidden: Tensor, mask: Tensor) -> Tensor:
        """Max-pool observed entries, with a zero vector for an empty set."""

        if hidden.shape[1] == 0:
            return hidden.new_zeros((hidden.shape[0], hidden.shape[2]))
        masked = hidden.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min)
        maximum = masked.max(dim=1).values
        return torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        option_features: Tensor,
        action_mask: Tensor,
        option_observation_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate a padded batch of graph states.

        Args:
            node_features: Node inputs with shape ``[B, N, F]``.
            adjacency: Directed weighted adjacency with shape ``[B, N, N]``.
            node_mask: Truthy entries identify real nodes, shape ``[B, N]``.
            global_features: State/player inputs with shape ``[B, G]``.
            option_features: Event-option inputs with shape ``[B, O, OF]``.
            action_mask: Truthy entries identify legal actions, shape
                ``[B, N + O]``.  Padded options must be false here.
            option_observation_mask: Displayed options with shape ``[B, O]``.
                Includes visible but currently illegal options; padding is false.
                If omitted, the option portion of ``action_mask`` is used.

        Returns:
            ``(masked_logits, value)`` with shapes ``[B, N + O]`` and ``[B]``.
            Illegal actions use the finite minimum of the output dtype.  If a
            row has no legal action, its logits are all zero so a downstream
            softmax remains finite; callers should treat that row as terminal.
        """

        batch_size, node_count, option_count = self._validate_inputs(
            node_features,
            adjacency,
            node_mask,
            global_features,
            option_features,
            action_mask,
            option_observation_mask,
        )

        device, dtype = self.device, self.dtype
        node_mask = node_mask.to(device=device, dtype=torch.bool)
        action_mask = action_mask.to(device=device, dtype=torch.bool)
        node_features = node_features.to(device=device, dtype=dtype)
        adjacency = adjacency.to(device=device, dtype=dtype)
        global_features = global_features.to(device=device, dtype=dtype)
        option_features = option_features.to(device=device, dtype=dtype)

        # Observation and legality differ: an unaffordable displayed purchase
        # can inform a sale decision, but padding can never become an action.
        option_mask = (
            action_mask[:, node_count:]
            if option_observation_mask is None
            else option_observation_mask.to(device=device, dtype=torch.bool)
        )
        effective_action_mask = torch.cat(
            (action_mask[:, :node_count] & node_mask, action_mask[:, node_count:] & option_mask),
            dim=1,
        )

        # Work only on the shared observed/selectable prefix of the menu. Keep
        # every node and every original option index, including unavailable
        # displayed purchases and gaps within the prefix. The full output is
        # restored below so Categorical retains its original sampling shape.
        full_option_count = option_count
        if option_count and not (self.training and self.config.dropout > 0):
            occupied = (option_mask | action_mask[:, node_count:]).any(dim=0)
            occupied_indices = occupied.nonzero(as_tuple=False)
            option_count = max(1, int(occupied_indices[-1, 0]) + 1 if len(occupied_indices) else 0)
            option_features = option_features[:, :option_count]
            option_mask = option_mask[:, :option_count]
        # During training with dropout, the original padded operations also
        # consume random numbers. Retaining them preserves that RNG behavior.

        clean_node_features = torch.where(
            node_mask.unsqueeze(-1),
            node_features,
            torch.zeros_like(node_features),
        )
        clean_option_features = torch.where(
            option_mask.unsqueeze(-1),
            option_features,
            torch.zeros_like(option_features),
        )

        valid_edges = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        clean_adjacency = torch.where(
            valid_edges, adjacency, torch.zeros_like(adjacency)
        )
        outgoing_adjacency = self._normalize_adjacency(clean_adjacency)
        incoming_adjacency = self._normalize_adjacency(
            clean_adjacency.transpose(1, 2)
        )

        hidden = self.node_encoder(clean_node_features)
        hidden = torch.where(
            node_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden)
        )
        for layer in self.message_passing:
            hidden = layer(
                hidden, outgoing_adjacency, incoming_adjacency, node_mask
            )

        graph_context = self._masked_mean(hidden, node_mask)
        global_context = self.global_encoder(global_features)
        option_hidden = self.option_encoder(clean_option_features)
        menu_context = self.menu_encoder(torch.cat(
            (self._masked_mean(option_hidden, option_mask),
             self._masked_max(option_hidden, option_mask)), dim=-1
        ))
        menu_context = torch.where(
            option_mask.any(dim=1, keepdim=True), menu_context, torch.zeros_like(menu_context)
        )

        expanded_graph_for_nodes = graph_context.unsqueeze(1).expand(
            batch_size, node_count, -1
        )
        expanded_global_for_nodes = global_context.unsqueeze(1).expand(
            batch_size, node_count, -1
        )
        node_policy_input = torch.cat(
            (hidden, expanded_graph_for_nodes, expanded_global_for_nodes,
             menu_context.unsqueeze(1).expand(batch_size, node_count, -1)), dim=-1
        )
        node_logits = self.node_policy_head(node_policy_input).squeeze(-1)

        expanded_graph_for_options = graph_context.unsqueeze(1).expand(
            batch_size, option_count, -1
        )
        expanded_global_for_options = global_context.unsqueeze(1).expand(
            batch_size, option_count, -1
        )
        option_policy_input = torch.cat(
            (
                option_hidden,
                expanded_graph_for_options,
                expanded_global_for_options,
                menu_context.unsqueeze(1).expand(batch_size, option_count, -1),
            ),
            dim=-1,
        )
        option_logits = self.option_policy_head(option_policy_input).squeeze(-1)

        if option_count < full_option_count:
            option_logits = F.pad(option_logits, (0, full_option_count - option_count),
                value=torch.finfo(option_logits.dtype).min)

        raw_logits = torch.cat((node_logits, option_logits), dim=1)
        mask_value = torch.finfo(raw_logits.dtype).min
        masked_logits = raw_logits.masked_fill(~effective_action_mask, mask_value)
        has_legal_action = effective_action_mask.any(dim=1, keepdim=True)
        masked_logits = torch.where(
            has_legal_action, masked_logits, torch.zeros_like(masked_logits)
        )

        value_input = torch.cat((graph_context, global_context, menu_context), dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        return masked_logits, value

    @torch.inference_mode()
    def predict(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        option_features: Tensor,
        action_mask: Tensor,
        option_observation_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run inference for one unbatched state.

        Inputs have shapes ``[N, F]``, ``[N, N]``, ``[N]``, ``[G]``,
        ``[O, OF]``, and ``[N + O]`` respectively.  The returned policy tensor
        has shape ``[N + O]`` and the value is a scalar tensor.  The module's
        previous training/evaluation mode is restored after the call.
        The optional observation mask is unbatched, with shape ``[O]``.
        """

        single_inputs = {
            "node_features": (node_features, 2),
            "adjacency": (adjacency, 2),
            "node_mask": (node_mask, 1),
            "global_features": (global_features, 1),
            "option_features": (option_features, 2),
            "action_mask": (action_mask, 1),
        }
        if option_observation_mask is not None:
            single_inputs["option_observation_mask"] = (option_observation_mask, 1)
        for name, (tensor, expected_rank) in single_inputs.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim != expected_rank:
                raise ValueError(
                    f"predict expected {name} to have rank {expected_rank}, "
                    f"got shape {tuple(tensor.shape)}"
                )

        was_training = self.training
        self.eval()
        try:
            logits, value = self(
                node_features.unsqueeze(0),
                adjacency.unsqueeze(0),
                node_mask.unsqueeze(0),
                global_features.unsqueeze(0),
                option_features.unsqueeze(0),
                action_mask.unsqueeze(0),
                None if option_observation_mask is None else option_observation_mask.unsqueeze(0),
            )
        finally:
            self.train(was_training)
        return logits.squeeze(0), value.squeeze(0)

    def save_checkpoint(
        self,
        path: PathType,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save architecture, weights, and optional training metadata."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata) if metadata is not None else {},
        }
        torch.save(checkpoint, destination)

    @classmethod
    def load_checkpoint(
        cls,
        path: PathType,
        map_location: str | torch.device | Mapping[str, str] | None = None,
        *,
        strict: bool = True,
    ) -> tuple[GraphPolicyValueNetwork, dict[str, Any]]:
        """Load a saved model and return ``(model, metadata)``.

        Checkpoints use Python pickle internally, as standard ``torch.save``
        files do.  Only load files from trusted sources.
        """

        try:
            checkpoint = torch.load(
                Path(path), map_location=map_location, weights_only=False
            )
        except TypeError:  # Compatibility with older PyTorch releases.
            checkpoint = torch.load(Path(path), map_location=map_location)

        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must contain a mapping")
        version = checkpoint.get("format_version")
        if version != cls.CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version {version!r}; "
                f"expected {cls.CHECKPOINT_FORMAT_VERSION}"
            )

        config_data = checkpoint.get("config")
        state_dict = checkpoint.get("state_dict")
        metadata = checkpoint.get("metadata", {})
        if not isinstance(config_data, Mapping):
            raise ValueError("checkpoint field 'config' must be a mapping")
        if config_data.get("architecture_version") != 2:
            raise ValueError(
                "checkpoint config must explicitly declare architecture_version 2; "
                "older policy/value heads are incompatible"
            )
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint field 'state_dict' must be a mapping")
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint field 'metadata' must be a mapping")

        model = cls(NetworkConfig(**dict(config_data)))
        # Constructing a module defaults to CPU/float32.  Match the loaded state
        # before copying so checkpoint dtype and ``map_location`` are respected.
        reference_tensor = next(
            (
                tensor
                for tensor in state_dict.values()
                if isinstance(tensor, Tensor) and tensor.is_floating_point()
            ),
            None,
        )
        if reference_tensor is not None:
            model.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        model.load_state_dict(state_dict, strict=strict)
        return model, dict(metadata)


# A concise alias is useful to callers that do not need to emphasize the graph
# implementation detail.  It is the exact same class, not a wrapper/subclass.
PolicyValueNetwork = GraphPolicyValueNetwork


__all__ = ["GraphPolicyValueNetwork", "NetworkConfig", "PolicyValueNetwork"]
