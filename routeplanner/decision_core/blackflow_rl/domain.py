from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping


class NodeType(str, Enum):
    """Client node types for ``rogue_6`` plus the simulator-only start node."""

    START = "START"
    BATTLE_NORMAL = "BATTLE_NORMAL"
    BATTLE_ELITE = "BATTLE_ELITE"
    BATTLE_BOSS = "BATTLE_BOSS"
    BATTLE_SHOP = "BATTLE_SHOP"
    REST = "REST"
    INCIDENT = "INCIDENT"
    WISH = "WISH"
    SACRIFICE = "SACRIFICE"
    EXPEDITION = "EXPEDITION"
    PORTAL = "PORTAL"
    DUEL = "DUEL"
    STORY = "STORY"
    STORY_HIDDEN = "STORY_HIDDEN"
    SCRAP_SHOP = "SCRAP_SHOP"
    DOOR = "DOOR"
    FINAL = "FINAL"
    EVACUATE = "EVACUATE"
    EMPLOY = "EMPLOY"
    LIGHT = "LIGHT"
    BATTLE_SAVAGE = "BATTLE_SAVAGE"
    EMPTY = "EMPTY"


BATTLE_TYPES = frozenset(
    {
        NodeType.BATTLE_NORMAL,
        NodeType.BATTLE_ELITE,
        NodeType.BATTLE_BOSS,
        NodeType.BATTLE_SAVAGE,
    }
)
EXIT_TYPES = frozenset({NodeType.FINAL, NodeType.EVACUATE, NodeType.BATTLE_BOSS})


class ActionKind(str, Enum):
    MOVE = "MOVE"
    CHOOSE = "CHOOSE"


@dataclass(frozen=True, slots=True)
class ResourceDelta:
    hp: int = 0
    max_hp: int = 0
    shield: int = 0
    gold: int = 0
    hope: int = 0
    parts: int = 0
    relics: int = 0
    tickets: int = 0
    team_strength: int = 0
    action_points: int = 0

    def __add__(self, other: "ResourceDelta") -> "ResourceDelta":
        return ResourceDelta(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class ResourceState:
    hp: int = 8
    max_hp: int = 8
    shield: int = 0
    gold: int = 8
    hope: int = 6
    parts: int = 0
    relics: int = 0
    tickets: int = 0
    team_strength: int = 6
    action_points: int = 0

    def apply(self, delta: ResourceDelta) -> "ResourceState":
        max_hp = max(1, self.max_hp + delta.max_hp)
        return ResourceState(
            hp=max(0, min(max_hp, self.hp + delta.hp + max(0, delta.max_hp))),
            max_hp=max_hp,
            shield=max(0, self.shield + delta.shield),
            gold=max(0, self.gold + delta.gold),
            hope=max(0, self.hope + delta.hope),
            parts=max(0, self.parts + delta.parts),
            relics=max(0, self.relics + delta.relics),
            tickets=max(0, self.tickets + delta.tickets),
            team_strength=max(0, self.team_strength + delta.team_strength),
            action_points=max(0, self.action_points + delta.action_points),
        )

    def can_apply(self, delta: ResourceDelta) -> bool:
        return all(
            getattr(self, name) + getattr(delta, name) >= 0
            for name in (
                "gold",
                "hope",
                "parts",
                "relics",
                "tickets",
                "team_strength",
                "action_points",
            )
        ) and self.hp + delta.hp > 0


@dataclass(frozen=True, slots=True)
class EventOption:
    option_id: str
    title: str
    effect: ResourceDelta = ResourceDelta()
    add_items: tuple[str, ...] = ()
    remove_items: tuple[str, ...] = ()
    battle: bool = False
    description: str = ""

    def is_available(self, resources: ResourceState, inventory: frozenset[str]) -> bool:
        return resources.can_apply(self.effect) and all(item in inventory for item in self.remove_items)


@dataclass(frozen=True, slots=True)
class MapNode:
    node_id: str
    index: int
    row: int
    col: int
    node_type: NodeType
    distance_from_start: int
    options: tuple[EventOption, ...] = ()
    auto_effect: ResourceDelta = ResourceDelta()
    event_name: str | None = None
    repeatable: bool = False
    requires_observation: bool = False

    @property
    def is_battle(self) -> bool:
        return self.node_type in BATTLE_TYPES

    @property
    def is_exit(self) -> bool:
        return self.node_type in EXIT_TYPES

    @property
    def hidden_category(self) -> str:
        return "ferocity" if self.is_battle else "mystery"


@dataclass(frozen=True, slots=True)
class FloorMap:
    floor: int
    width: int
    height: int
    nodes: tuple[MapNode, ...]
    edges: tuple[tuple[str, str], ...]
    start_node_id: str
    exit_node_ids: tuple[str, ...]
    seed: int
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            payload = {
                "floor": self.floor,
                "nodes": [
                    (n.node_id, n.row, n.col, n.node_type.value) for n in self.nodes
                ],
                "edges": sorted(tuple(sorted(edge)) for edge in self.edges),
                "seed": self.seed,
            }
            object.__setattr__(
                self,
                "fingerprint",
                sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16],
            )

    def node(self, node_id: str) -> MapNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def node_by_index(self, index: int) -> MapNode:
        if index < 0 or index >= len(self.nodes):
            raise IndexError(index)
        node = self.nodes[index]
        if node.index != index:
            raise ValueError("node indices must be contiguous and ordered")
        return node

    def adjacency(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for left, right in self.edges:
            result[left].append(right)
            result[right].append(left)
        return {key: tuple(sorted(value)) for key, value in result.items()}


@dataclass(frozen=True, slots=True)
class Action:
    action_id: int
    kind: ActionKind
    target_node_id: str | None = None
    option_index: int | None = None
    movement_cost: int = 0


@dataclass(frozen=True, slots=True)
class GameState:
    maps: tuple[FloorMap, ...]
    floor_index: int
    current_node_id: str
    resources: ResourceState
    completed: frozenset[str]
    revealed: frozenset[str]
    inventory: frozenset[str] = frozenset()
    pending_node_id: str | None = None
    terminal: bool = False
    total_reward: float = 0.0
    step_count: int = 0
    chase_count: int = 0
    history: tuple[str, ...] = ()

    @property
    def floor_map(self) -> FloorMap:
        return self.maps[self.floor_index]

    @property
    def floor(self) -> int:
        return self.floor_map.floor

    def with_reward(self, reward: float, message: str) -> "GameState":
        return replace(
            self,
            total_reward=self.total_reward + reward,
            step_count=self.step_count + 1,
            history=self.history + (message,),
        )

    def state_key(self) -> tuple[Any, ...]:
        """Complete key suitable for diagnostics or future transposition tables."""

        r = self.resources
        return (
            tuple(m.fingerprint for m in self.maps),
            self.floor_index,
            self.current_node_id,
            self.pending_node_id,
            self.completed,
            self.revealed,
            self.inventory,
            tuple(getattr(r, name) for name in r.__dataclass_fields__),
            self.terminal,
        )


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: GameState
    reward: float
    terminated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
