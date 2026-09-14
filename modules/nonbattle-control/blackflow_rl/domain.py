from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .utopia_economy import RegionState
    from .residents import ResidentContext
    from .fate_nodes import FateContext
    from .notebook_observation import NotebookObservation
    from .recruitment import UnknownReserveOpportunity
    from .recruitment_candidates import RecruitTicketCandidateGroup
    from .temporary_recruitment import ObservedTemporaryOffer
    from .employment import EmploymentContext


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
    EQUIP = "EQUIP"
    DISCARD = "DISCARD"
    RETURN = "RETURN"


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
        hp_after_capacity=(max(1,self.hp+delta.max_hp) if delta.max_hp<0
                           else self.hp+max(0,delta.max_hp))
        return ResourceState(
            hp=max(0, min(max_hp, hp_after_capacity + delta.hp)),
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
    operation: str = "event"
    item_id: str | None = None
    instance_id: str | None = None
    quantity: int = 1
    price: int = 0
    ends_node: bool = True

    def is_available(self, resources: ResourceState, inventory: frozenset[str]) -> bool:
        return (
            resources.can_apply(self.effect)
            and all(item in inventory for item in self.remove_items)
            and not any(item in inventory for item in self.add_items)
        )


@dataclass(frozen=True, slots=True)
class ItemInstance:
    instance_id: str
    item_id: str
    category: str
    uses_remaining: int | None = None
    appraisal: int = 0


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    step: int
    floor: int
    node_id: str
    operation: str
    item_id: str | None = None
    instance_id: str | None = None
    category: str | None = None
    quantity: int = 0
    gold_delta: int = 0
    source: str = ""


@dataclass(frozen=True, slots=True)
class ShopStock:
    slot_id: str
    item_id: str
    price: int
    sold: bool = False


@dataclass(frozen=True, slots=True)
class ShopState:
    node_id: str
    stock: tuple[ShopStock, ...] = ()
    refreshes: int = 0
    sold_parts: int = 0
    trade_bonus_paid: bool = False
    visits: int = 0
    lost_slots: int = 0
    entry_withdrawn: int = 0
    total_withdrawn: int = 0
    last_entry_movement_count: int | None = None


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
    stage_id: str | None = None

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
    equipment_instance_id: str | None = None
    traversed_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PortalContext:
    """Suspended ordinary region while exploring one independent Black Pond."""

    zone_id: str
    variation_id: int
    outer_floor_map: FloorMap
    outer_node_id: str
    outer_action_points: int
    outer_completed: frozenset[str]
    outer_revealed: frozenset[str]
    outer_seen_event_names: frozenset[str]
    resident_node_ids: tuple[str, ...] = ()
    rewarded_node_ids: frozenset[str] = frozenset()
    exhausted_operator_ids: frozenset[str] = frozenset()
    outer_region_state: RegionState | None = None
    outer_resident_context: ResidentContext | None = None
    outer_fate_context: FateContext | None = None


@dataclass(frozen=True, slots=True)
class PendingExpedition:
    operator_id: str
    kind: str
    departure_floor: int
    return_floor_index: int


@dataclass(frozen=True, slots=True)
class ChaseRewardContext:
    """Fixed off-map loot; the current node only carries its visible menu."""

    occurrence_id: str
    floor: int
    battle_number: int
    carrier_node: MapNode
    relic_options: tuple[EventOption, ...] = ()
    part_options: tuple[EventOption, ...] = ()


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
    item_instances: tuple[ItemInstance, ...] = ()
    ledger: tuple[LedgerEntry, ...] = ()
    shops: tuple[ShopState, ...] = ()
    equipped_instance_id: str | None = None
    economy_enabled: bool = False
    economy_seed: int = 0
    random_counter: int = 0
    item_serial: int = 0
    parts_capacity: int = 8
    squad_capacity: int = 6
    supply_vouchers: int = 0
    event_counters: tuple[tuple[str, int], ...] = ()
    seen_event_names: frozenset[str] = frozenset()
    battle_count: int = 0
    movement_count: int = 0
    awaiting_exit: bool = False
    portal_context: PortalContext | None = None
    region_state: RegionState | None = None
    resident_context: ResidentContext | None = None
    fate_context: FateContext | None = None
    formal_operator_ids: frozenset[str] = frozenset()
    promoted_operator_ids: frozenset[str] = frozenset()
    expeditions: tuple[PendingExpedition, ...] = ()
    ending_id: str | None = None
    door_ready_node_id: str | None = None
    bank_initial_balance: int = 0
    bank_balance: int = 0
    total_bank_withdrawn: int = 0
    bank_balance_spent: int = 0
    notebook_observation: NotebookObservation | None = None
    pending_recruit_ticket_ids: tuple[str, ...] = ()
    stored_recruit_ticket_ids: tuple[str, ...] = ()
    unknown_recruit_opportunities: tuple[UnknownReserveOpportunity, ...] = ()
    unknown_recruit_serial: int = 0
    chase_reward_pending: bool = False
    pending_recruit_candidate_groups: tuple[RecruitTicketCandidateGroup, ...] = ()
    recruit_candidate_serial: int = 0
    temporary_recruit_offers: tuple[ObservedTemporaryOffer, ...] = ()
    employment_context: EmploymentContext | None = None
    chase_reward_context: ChaseRewardContext | None = None

    def __setstate__(self, values):
        # Diagnostic snapshots predate the appended off-map reward context.
        # Preserve their original field order instead of shifting old values
        # into the new slot; missing appended fields use their defaults.
        from dataclasses import fields, MISSING
        definitions=fields(self)
        if len(values)>len(definitions):
            raise ValueError('snapshot has an unsupported newer state schema')
        for index, definition in enumerate(definitions):
            if index<len(values):
                value=values[index]
            elif definition.default is not MISSING:
                value=definition.default
            elif definition.default_factory is not MISSING:
                value=definition.default_factory()
            else:
                raise ValueError('snapshot omits a required state field')
            object.__setattr__(self,definition.name,value)

    @property
    def available_formal_operator_ids(self) -> frozenset[str]:
        return self.formal_operator_ids - {x.operator_id for x in self.expeditions}

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
            self.item_instances,
            self.shops,
            self.equipped_instance_id,
            self.economy_enabled,
            self.economy_seed,
            self.random_counter,
            self.item_serial,
            self.parts_capacity,
            self.squad_capacity,
            self.supply_vouchers,
            self.event_counters,
            self.seen_event_names,
            self.battle_count,
            self.movement_count,
            self.awaiting_exit,
            self.portal_context,
            self.region_state,
            self.resident_context,
            self.fate_context,
            self.formal_operator_ids,
            self.promoted_operator_ids,
            self.expeditions,
            self.ending_id,
            self.door_ready_node_id,
            self.bank_initial_balance,
            self.bank_balance,
            self.total_bank_withdrawn,
            self.bank_balance_spent,
            self.notebook_observation,
            self.pending_recruit_ticket_ids,
            self.stored_recruit_ticket_ids,
            self.unknown_recruit_opportunities,
            self.unknown_recruit_serial,
            self.chase_reward_pending,
            self.chase_reward_context,
            self.pending_recruit_candidate_groups,
            self.recruit_candidate_serial,
            self.temporary_recruit_offers,
            self.employment_context,
        )


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: GameState
    reward: float
    terminated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
