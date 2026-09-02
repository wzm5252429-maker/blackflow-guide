from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .domain import FloorMap, NodeType


Slot = tuple[int, int]
Edge = tuple[Slot, Slot]

DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "rules"
    / "blackflow_map_templates_v1.json"
)
EXPECTED_TEMPLATE_COUNTS: Mapping[int, int] = MappingProxyType(
    {1: 3, 2: 10, 3: 10, 4: 10, 5: 10, 6: 1}
)


class TemplateValidationError(ValueError):
    """Raised when the checked-in topology catalogue violates its schema."""


def _canonical_edge(left: Slot, right: Slot) -> Edge:
    return (left, right) if left < right else (right, left)


@dataclass(frozen=True, slots=True)
class FixedSlot:
    slot: Slot
    node_type: NodeType


@dataclass(frozen=True, slots=True)
class MapTemplate:
    """One immutable, fully validated Blackflow map topology.

    ``edges`` are stored as sorted, canonical undirected edges.  The occupied
    slots are derived rather than trusted from an additional input field, so a
    catalogue entry cannot silently disagree with its graph.
    """

    template_id: str
    floor: int
    rows: int
    cols: int
    start: Slot
    final_slots: tuple[Slot, ...]
    boss_terminal: Slot | None
    edges: tuple[Edge, ...]
    fixed_slots: tuple[FixedSlot, ...]
    known_weight: float | None
    fallback_weight: float
    notes: str = ""
    occupied_slots: frozenset[Slot] = field(init=False)

    def __post_init__(self) -> None:
        if not self.template_id:
            raise TemplateValidationError("template_id must be non-empty")
        if self.floor not in EXPECTED_TEMPLATE_COUNTS:
            raise TemplateValidationError(
                f"{self.template_id}: floor must be one of "
                f"{tuple(EXPECTED_TEMPLATE_COUNTS)}"
            )
        if self.rows <= 0 or self.cols <= 0:
            raise TemplateValidationError(
                f"{self.template_id}: rows and cols must be positive"
            )

        normalized_edges = tuple(
            sorted(_canonical_edge(left, right) for left, right in self.edges)
        )
        if len(set(normalized_edges)) != len(normalized_edges):
            raise TemplateValidationError(
                f"{self.template_id}: duplicate undirected edge"
            )
        object.__setattr__(self, "edges", normalized_edges)

        edge_slots: set[Slot] = set()
        for left, right in normalized_edges:
            if left == right:
                raise TemplateValidationError(
                    f"{self.template_id}: self-loop at {left}"
                )
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) != 1:
                raise TemplateValidationError(
                    f"{self.template_id}: non-orthogonal edge {left!r}-{right!r}"
                )
            edge_slots.update((left, right))

        terminals = set(self.final_slots)
        if len(terminals) != len(self.final_slots):
            raise TemplateValidationError(
                f"{self.template_id}: duplicate FINAL slot"
            )
        if self.boss_terminal is not None and self.final_slots:
            raise TemplateValidationError(
                f"{self.template_id}: boss and FINAL slots are mutually exclusive"
            )
        if self.floor in (1, 2, 4):
            if not self.final_slots or self.boss_terminal is not None:
                raise TemplateValidationError(
                    f"{self.template_id}: floor {self.floor} requires FINAL slots"
                )
        else:
            if self.boss_terminal is None or self.final_slots:
                raise TemplateValidationError(
                    f"{self.template_id}: floor {self.floor} requires one boss terminal"
                )

        fixed_positions = [item.slot for item in self.fixed_slots]
        if len(set(fixed_positions)) != len(fixed_positions):
            raise TemplateValidationError(
                f"{self.template_id}: duplicate fixed slot"
            )

        declared_slots = {self.start, *self.final_slots, *fixed_positions}
        if self.boss_terminal is not None:
            declared_slots.add(self.boss_terminal)
        occupied = frozenset(edge_slots | declared_slots)
        object.__setattr__(self, "occupied_slots", occupied)

        for row, col in occupied:
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise TemplateValidationError(
                    f"{self.template_id}: slot {(row, col)} is outside "
                    f"{self.rows}x{self.cols} bounds"
                )

        terminal_slots = set(self.final_slots)
        if self.boss_terminal is not None:
            terminal_slots.add(self.boss_terminal)
        if self.start in terminal_slots:
            raise TemplateValidationError(
                f"{self.template_id}: start cannot also be a terminal"
            )
        if self.start in fixed_positions or terminal_slots.intersection(fixed_positions):
            raise TemplateValidationError(
                f"{self.template_id}: fixed slots cannot overlap start or terminals"
            )

        # All declared locations must participate in the topology rather than
        # becoming isolated vertices merely because they were declared.
        missing_from_edges = declared_slots - edge_slots
        if missing_from_edges:
            raise TemplateValidationError(
                f"{self.template_id}: declared slots absent from edges: "
                f"{sorted(missing_from_edges)!r}"
            )

        adjacency: dict[Slot, set[Slot]] = {slot: set() for slot in occupied}
        for left, right in normalized_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        reached = {self.start}
        queue: deque[Slot] = deque((self.start,))
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current] - reached:
                reached.add(neighbor)
                queue.append(neighbor)
        if reached != set(occupied):
            raise TemplateValidationError(
                f"{self.template_id}: topology is disconnected"
            )

        if self.known_weight is not None and (
            not isfinite(self.known_weight) or self.known_weight <= 0
        ):
            raise TemplateValidationError(
                f"{self.template_id}: known_weight must be positive or null"
            )
        if not isfinite(self.fallback_weight) or self.fallback_weight <= 0:
            raise TemplateValidationError(
                f"{self.template_id}: fallback_weight must be positive"
            )

    @property
    def terminal_slots(self) -> tuple[Slot, ...]:
        if self.boss_terminal is not None:
            return (self.boss_terminal,)
        return self.final_slots

    @property
    def selection_weight(self) -> float:
        """Observed weight when known, otherwise the documented fallback prior."""

        return (
            self.known_weight
            if self.known_weight is not None
            else self.fallback_weight
        )

    @property
    def fixed_slot_types(self) -> Mapping[Slot, NodeType]:
        return MappingProxyType(
            {item.slot: item.node_type for item in self.fixed_slots}
        )

    def topology_matches(self, floor_map: FloorMap) -> bool:
        """Return whether a generated map exactly uses this template layout.

        Layer five contains one nine-column template while the simulator's
        floor envelope is ten columns wide, so map dimensions may be larger
        than the template but never smaller.
        """

        if (
            floor_map.floor != self.floor
            or floor_map.height < self.rows
            or floor_map.width < self.cols
        ):
            return False

        positions = {node.node_id: (node.row, node.col) for node in floor_map.nodes}
        if len(positions) != len(floor_map.nodes):
            return False
        if len(set(positions.values())) != len(positions):
            return False
        if set(positions.values()) != set(self.occupied_slots):
            return False
        if positions.get(floor_map.start_node_id) != self.start:
            return False

        try:
            actual_edges = tuple(
                sorted(
                    _canonical_edge(positions[left], positions[right])
                    for left, right in floor_map.edges
                )
            )
        except KeyError:
            return False
        return actual_edges == self.edges


@dataclass(frozen=True, slots=True)
class TemplateLibrary:
    schema_version: int
    source_snapshot: str
    source_page: str
    source_map: str
    weight_status: str
    templates: tuple[MapTemplate, ...]
    _by_floor: Mapping[int, tuple[MapTemplate, ...]] = field(
        init=False, repr=False, compare=False
    )
    _by_id: Mapping[str, MapTemplate] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise TemplateValidationError(
                f"unsupported map template schema_version {self.schema_version!r}"
            )
        if len(self.templates) != sum(EXPECTED_TEMPLATE_COUNTS.values()):
            raise TemplateValidationError(
                f"expected {sum(EXPECTED_TEMPLATE_COUNTS.values())} templates, "
                f"got {len(self.templates)}"
            )

        ids = [item.template_id for item in self.templates]
        if len(set(ids)) != len(ids):
            duplicates = sorted(
                template_id
                for template_id, count in Counter(ids).items()
                if count > 1
            )
            raise TemplateValidationError(
                f"duplicate template ids: {duplicates!r}"
            )

        by_floor = {
            floor: tuple(
                item for item in self.templates if item.floor == floor
            )
            for floor in EXPECTED_TEMPLATE_COUNTS
        }
        actual_counts = {floor: len(items) for floor, items in by_floor.items()}
        if actual_counts != dict(EXPECTED_TEMPLATE_COUNTS):
            raise TemplateValidationError(
                f"template counts by floor must be "
                f"{dict(EXPECTED_TEMPLATE_COUNTS)!r}, got {actual_counts!r}"
            )

        object.__setattr__(self, "_by_floor", MappingProxyType(by_floor))
        object.__setattr__(
            self,
            "_by_id",
            MappingProxyType({item.template_id: item for item in self.templates}),
        )

    def for_floor(self, floor: int) -> tuple[MapTemplate, ...]:
        try:
            return self._by_floor[floor]
        except KeyError as exc:
            raise KeyError(f"no templates for floor {floor}") from exc

    def get(self, template_id: str) -> MapTemplate:
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown map template {template_id!r}") from exc

    def selection_weights(self, floor: int) -> tuple[float, ...]:
        return tuple(item.selection_weight for item in self.for_floor(floor))

    def find_match(self, floor_map: FloorMap) -> MapTemplate | None:
        matches = [
            item
            for item in self.for_floor(floor_map.floor)
            if item.topology_matches(floor_map)
        ]
        if len(matches) > 1:
            raise TemplateValidationError(
                f"map matches multiple templates: "
                f"{[item.template_id for item in matches]!r}"
            )
        return matches[0] if matches else None


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise TemplateValidationError(f"{context}: missing field {key!r}")
    return mapping[key]


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemplateValidationError(f"{context}: expected integer")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TemplateValidationError(f"{context}: expected string")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemplateValidationError(f"{context}: expected number")
    result = float(value)
    if not isfinite(result):
        raise TemplateValidationError(f"{context}: expected finite number")
    return result


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TemplateValidationError(f"{context}: expected array")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TemplateValidationError(f"{context}: expected object")
    return value


def _slot(value: Any, context: str) -> Slot:
    values = _sequence(value, context)
    if len(values) != 2:
        raise TemplateValidationError(f"{context}: coordinate must have two items")
    return (
        _integer(values[0], f"{context}[0]"),
        _integer(values[1], f"{context}[1]"),
    )


def _parse_template(value: Any, index: int) -> MapTemplate:
    raw = _mapping(value, f"templates[{index}]")
    context = f"template {raw.get('template_id', index)!r}"

    raw_edges = _sequence(_required(raw, "edges", context), f"{context}.edges")
    edges: list[Edge] = []
    for edge_index, value in enumerate(raw_edges):
        pair = _sequence(value, f"{context}.edges[{edge_index}]")
        if len(pair) != 2:
            raise TemplateValidationError(
                f"{context}.edges[{edge_index}]: edge must have two endpoints"
            )
        edges.append(
            (
                _slot(pair[0], f"{context}.edges[{edge_index}][0]"),
                _slot(pair[1], f"{context}.edges[{edge_index}][1]"),
            )
        )

    raw_fixed = _sequence(
        _required(raw, "fixed_slots", context), f"{context}.fixed_slots"
    )
    fixed_slots: list[FixedSlot] = []
    for fixed_index, value in enumerate(raw_fixed):
        item = _mapping(value, f"{context}.fixed_slots[{fixed_index}]")
        type_value = _string(
            _required(item, "node_type", context),
            f"{context}.fixed_slots[{fixed_index}].node_type",
        )
        try:
            node_type = NodeType(type_value)
        except ValueError as exc:
            raise TemplateValidationError(
                f"{context}.fixed_slots[{fixed_index}]: unknown node type "
                f"{type_value!r}"
            ) from exc
        fixed_slots.append(
            FixedSlot(
                slot=_slot(
                    _required(item, "slot", context),
                    f"{context}.fixed_slots[{fixed_index}].slot",
                ),
                node_type=node_type,
            )
        )

    final_slots = tuple(
        _slot(item, f"{context}.final_slots[{terminal_index}]")
        for terminal_index, item in enumerate(
            _sequence(
                _required(raw, "final_slots", context),
                f"{context}.final_slots",
            )
        )
    )
    boss_value = _required(raw, "boss_terminal", context)
    boss_terminal = (
        None if boss_value is None else _slot(boss_value, f"{context}.boss_terminal")
    )
    known_value = _required(raw, "known_weight", context)
    known_weight = (
        None if known_value is None else _number(known_value, f"{context}.known_weight")
    )

    return MapTemplate(
        template_id=_string(
            _required(raw, "template_id", context), f"{context}.template_id"
        ),
        floor=_integer(_required(raw, "floor", context), f"{context}.floor"),
        rows=_integer(_required(raw, "rows", context), f"{context}.rows"),
        cols=_integer(_required(raw, "cols", context), f"{context}.cols"),
        start=_slot(_required(raw, "start", context), f"{context}.start"),
        final_slots=final_slots,
        boss_terminal=boss_terminal,
        edges=tuple(edges),
        fixed_slots=tuple(fixed_slots),
        known_weight=known_weight,
        fallback_weight=_number(
            _required(raw, "fallback_weight", context),
            f"{context}.fallback_weight",
        ),
        notes=_string(raw.get("notes", ""), f"{context}.notes"),
    )


def load_template_library(
    path: str | Path | None = None,
) -> TemplateLibrary:
    """Load and validate the 43 normal templates plus the fixed VI template."""

    source_path = DEFAULT_TEMPLATE_PATH if path is None else Path(path)
    try:
        raw_value = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemplateValidationError(
            f"could not load map templates from {source_path}: {exc}"
        ) from exc
    raw = _mapping(raw_value, "template catalogue")
    templates = tuple(
        _parse_template(item, index)
        for index, item in enumerate(
            _sequence(_required(raw, "templates", "catalogue"), "catalogue.templates")
        )
    )
    return TemplateLibrary(
        schema_version=_integer(
            _required(raw, "schema_version", "catalogue"),
            "catalogue.schema_version",
        ),
        source_snapshot=_string(
            _required(raw, "source_snapshot", "catalogue"),
            "catalogue.source_snapshot",
        ),
        source_page=_string(
            _required(raw, "source_page", "catalogue"), "catalogue.source_page"
        ),
        source_map=_string(
            _required(raw, "source_map", "catalogue"), "catalogue.source_map"
        ),
        weight_status=_string(
            _required(raw, "weight_status", "catalogue"), "catalogue.weight_status"
        ),
        templates=templates,
    )


__all__ = [
    "DEFAULT_TEMPLATE_PATH",
    "EXPECTED_TEMPLATE_COUNTS",
    "Edge",
    "FixedSlot",
    "MapTemplate",
    "Slot",
    "TemplateLibrary",
    "TemplateValidationError",
    "load_template_library",
]
