from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .domain import NodeType, ResourceState


DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "rules" / "blackflow_sim_v1.json"
)


@dataclass(frozen=True, slots=True)
class FloorRule:
    floor: int
    width: int
    height: int
    node_count: tuple[int, int]
    action_points: int
    minimum_exit_distance: int
    maximum_graph_distance: int
    exit_types: tuple[NodeType, ...]


@dataclass(frozen=True, slots=True)
class NodeRule:
    node_type: NodeType
    label: str
    category: str
    weight: float
    counts: Mapping[int, tuple[int, int]]
    distances: Mapping[int, tuple[int, int]]

    def count_range(self, floor: int) -> tuple[int, int]:
        return self.counts.get(floor, (0, 0))

    def distance_range(self, floor: int) -> tuple[int, int] | None:
        return self.distances.get(floor)

    def allows(self, floor: int, distance: int) -> bool:
        distance_range = self.distance_range(floor)
        return bool(
            self.count_range(floor)[1] > 0
            and distance_range is not None
            and distance_range[0] <= distance <= distance_range[1]
        )


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    discount: float
    value_scale: float
    floor_clear_bonus: float
    run_clear_bonus: float
    chase_penalty: float
    weights: Mapping[str, float]

    def resource_reward(
        self,
        before: ResourceState,
        after: ResourceState,
        *,
        key_items_added: int = 0,
    ) -> float:
        reward = 0.0
        for name, weight in self.weights.items():
            if name == "key_item":
                reward += weight * key_items_added
            elif hasattr(before, name):
                reward += weight * (getattr(after, name) - getattr(before, name))
        return float(reward)


@dataclass(frozen=True, slots=True)
class Ruleset:
    schema_version: int
    ruleset_id: str
    title: str
    model_scope: str
    notes: tuple[str, ...]
    sources: tuple[str, ...]
    max_nodes: int
    max_options: int
    node_distance_metric: str
    floors: Mapping[int, FloorRule]
    node_rules: Mapping[NodeType, NodeRule]
    event_pools: Mapping[int, tuple[str, ...]]
    objective: ObjectiveConfig
    sha256: str
    source_path: Path
    map_templates_path: Path | None

    @property
    def action_size(self) -> int:
        return self.max_nodes + self.max_options

    def floor(self, floor: int) -> FloorRule:
        try:
            return self.floors[floor]
        except KeyError as exc:
            raise ValueError(f"ruleset does not define floor {floor}") from exc


def _pair(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element array")
    low, high = int(value[0]), int(value[1])
    if low < 0 or high < low:
        raise ValueError(f"invalid range for {field}: {value!r}")
    return low, high


def load_ruleset(path: str | Path | None = None) -> Ruleset:
    source_path = Path(path) if path is not None else DEFAULT_RULES_PATH
    source_path = source_path.resolve()
    raw_bytes = source_path.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))
    template_name = data.get("map_templates")
    map_templates_path = (
        (source_path.parent / str(template_name)).resolve()
        if template_name is not None
        else None
    )
    if map_templates_path is not None and not map_templates_path.is_file():
        raise ValueError(f"map template catalog does not exist: {map_templates_path}")
    fingerprint_bytes = raw_bytes
    if map_templates_path is not None:
        fingerprint_bytes += b"\0map-templates\0" + map_templates_path.read_bytes()

    schema_version = int(data.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"unsupported rules schema: {schema_version}")

    floors: dict[int, FloorRule] = {}
    for floor_text, item in data["floors"].items():
        floor = int(floor_text)
        floors[floor] = FloorRule(
            floor=floor,
            width=int(item["width"]),
            height=int(item["height"]),
            node_count=_pair(item["node_count"], field=f"floors.{floor}.node_count"),
            action_points=int(item["action_points"]),
            minimum_exit_distance=int(item["minimum_exit_distance"]),
            maximum_graph_distance=int(item["maximum_graph_distance"]),
            exit_types=tuple(NodeType(value) for value in item["exit_types"]),
        )

    node_rules: dict[NodeType, NodeRule] = {}
    for type_text, item in data["node_rules"].items():
        node_type = NodeType(type_text)
        counts = {
            int(floor): _pair(value, field=f"{type_text}.counts.{floor}")
            for floor, value in item.get("counts", {}).items()
        }
        distances = {
            int(floor): _pair(value, field=f"{type_text}.distances.{floor}")
            for floor, value in item.get("distances", {}).items()
        }
        if set(counts) != set(distances):
            raise ValueError(f"{type_text}: count and distance floors must match")
        node_rules[node_type] = NodeRule(
            node_type=node_type,
            label=str(item["label"]),
            category=str(item["category"]),
            weight=float(item["weight"]),
            counts=counts,
            distances=distances,
        )

    required = set(NodeType) - {NodeType.START}
    missing = required - set(node_rules)
    if missing:
        raise ValueError(f"missing node rules: {sorted(item.value for item in missing)}")

    objective_data = data["objective"]
    objective = ObjectiveConfig(
        discount=float(objective_data["discount"]),
        value_scale=float(objective_data["value_scale"]),
        floor_clear_bonus=float(objective_data["floor_clear_bonus"]),
        run_clear_bonus=float(objective_data["run_clear_bonus"]),
        chase_penalty=float(objective_data["chase_penalty"]),
        weights={str(key): float(value) for key, value in objective_data["weights"].items()},
    )
    if not 0 < objective.discount <= 1:
        raise ValueError("objective.discount must be in (0, 1]")
    if objective.value_scale <= 0:
        raise ValueError("objective.value_scale must be positive")

    max_nodes = int(data["max_nodes"])
    if max(rule.width * rule.height for rule in floors.values()) > max_nodes:
        raise ValueError("max_nodes is smaller than a configured floor grid")

    return Ruleset(
        schema_version=schema_version,
        ruleset_id=str(data["ruleset_id"]),
        title=str(data["title"]),
        model_scope=str(data["model_scope"]),
        notes=tuple(map(str, data.get("notes", []))),
        sources=tuple(map(str, data.get("sources", []))),
        max_nodes=max_nodes,
        max_options=int(data["max_options"]),
        node_distance_metric=str(data.get("node_distance_metric", "graph-bfs")),
        floors=floors,
        node_rules=node_rules,
        event_pools={int(key): tuple(map(str, value)) for key, value in data["event_pools"].items()},
        objective=objective,
        sha256=sha256(fingerprint_bytes).hexdigest(),
        source_path=source_path,
        map_templates_path=map_templates_path,
    )
