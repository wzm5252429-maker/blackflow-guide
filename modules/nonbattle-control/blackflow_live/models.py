"""Screenshot-grounded contracts shared by recognition, policy and execution.

Bounding boxes are (x, y, width, height) in the supplied recognition image's
pixels. They are never desktop coordinates. Missing observations stay missing;
this module does not fill them with the simulator's initial state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ObservedNode:
    node_id: str
    node_type: str
    row: int
    col: int
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    revealed: bool = True
    completed: bool = False


@dataclass(frozen=True, slots=True)
class ObservedAction:
    action_id: str
    label: str
    kind: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    enabled: bool = True
    target_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveObservation:
    frame_id: str
    captured_at: float
    scene: str
    confidence: float
    actions: tuple[ObservedAction, ...] = ()
    nodes: tuple[ObservedNode, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()
    resources: dict[str, float | int | None] = field(default_factory=dict)
    current_node_id: str | None = None
    floor: int | None = None
    ending_first_confirmed: bool = False
    diagnostics: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: ObservedAction | None
    reason: str
    policy_name: str = "current_neural_controller"
    neural: bool = False
    confidence: float = 0.0

    @property
    def action_id(self) -> str | None:
        return self.action.action_id if self.action is not None else None
