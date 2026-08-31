from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random
from typing import Iterable, Mapping

from .domain import EventOption, FloorMap, MapNode, NodeType, ResourceDelta
from .map_templates import MapTemplate, TemplateLibrary, load_template_library
from .rules import FloorRule, NodeRule, Ruleset, load_ruleset


GridPoint = tuple[int, int]


_CONTENT_TOTAL_RANGES: Mapping[int, Mapping[str, tuple[int, int]]] = {
    1: {"mystery": (1, 6), "ferocity": (2, 6)},
    2: {"mystery": (3, 9), "ferocity": (1, 7)},
    3: {"mystery": (7, 13), "ferocity": (4, 7)},
    4: {"mystery": (8, 14), "ferocity": (4, 8)},
    5: {"mystery": (9, 18), "ferocity": (5, 11)},
    # VI has one fixed, fully revealed topology.  The route tool proves four
    # content slots; the other eight use an explicit synthetic prior until
    # real run observations can replace these broad category bounds.
    6: {"mystery": (3, 9), "ferocity": (1, 6)},
}
_TOTAL_EXCLUDED_TYPES = frozenset(
    {
        NodeType.START,
        NodeType.EMPTY,
        NodeType.FINAL,
        NodeType.EVACUATE,
        NodeType.BATTLE_BOSS,
    }
)
_SPECIAL_FALLBACK_WEIGHTS: Mapping[NodeType, float] = {
    NodeType.DOOR: 0.55,
    NodeType.STORY: 1.0,
    NodeType.BATTLE_SAVAGE: 0.45,
}


class MapGenerationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MapGeneratorConfig:
    max_attempts: int = 200
    include_advanced_nodes: bool = False
    enable_portal: bool = False
    enable_expedition: bool = False
    enable_second_ending: bool = False
    enable_third_ending: bool = False
    door_pair_probability: float = 0.25
    evacuation_probability: float = 0.35


class MapGenerator:
    """Sample a verified topology, then solve its node-content constraints.

    The public tools expose 43 I--V main-route topologies plus one fixed VI
    third-ending topology.  The checked-in catalogue keeps those structures
    exact; synthetic fallback weights apply only to the unknown I--V sampling
    stages.  VI keeps its topology and four observed content slots fixed;
    only the eight source-unspecified slots use the versioned synthetic prior.
    """

    def __init__(
        self,
        ruleset: Ruleset | None = None,
        config: MapGeneratorConfig | None = None,
        templates: TemplateLibrary | None = None,
    ) -> None:
        self.ruleset = ruleset or load_ruleset()
        self.config = config or MapGeneratorConfig()
        self.templates = templates or load_template_library(
            self.ruleset.map_templates_path
        )
        if not 0 <= self.config.door_pair_probability <= 1:
            raise ValueError("door_pair_probability must be in [0, 1]")
        if not 0 <= self.config.evacuation_probability <= 1:
            raise ValueError("evacuation_probability must be in [0, 1]")
        if self.config.enable_second_ending and self.config.enable_third_ending:
            raise ValueError(
                "enable_second_ending and enable_third_ending are mutually exclusive"
            )

    def generate_run(self, seed: int) -> tuple[FloorMap, ...]:
        master = random.Random(seed)
        floors = (
            floor
            for floor in sorted(self.ruleset.floors)
            if floor != 6 or self.config.enable_third_ending
        )
        return tuple(
            self.generate_floor(floor, master.getrandbits(63))
            for floor in floors
        )

    def generate_floor(
        self, floor: int, seed: int, *, template_id: str | None = None
    ) -> FloorMap:
        floor_rule = self.ruleset.floor(floor)
        master = random.Random(seed)
        if template_id is None:
            candidates = self.templates.for_floor(floor)
            template = master.choices(
                candidates,
                weights=self.templates.selection_weights(floor),
                k=1,
            )[0]
        else:
            template = self.templates.get(template_id)
            if template.floor != floor:
                raise ValueError(
                    f"template {template_id!r} belongs to floor {template.floor}, not {floor}"
                )
        door_count = (
            2
            if floor in (3, 4, 5)
            and master.random() < self.config.door_pair_probability
            else 0
        )
        story_count = (
            1
            if floor == 6
            else 3
            if floor == 5 and self.config.enable_second_ending
            else 0
        )
        evacuation_count = (
            1
            if floor in (1, 2, 4)
            and master.random() < self.config.evacuation_probability
            else 0
        )
        last_error = "unknown failure"
        for _ in range(self.config.max_attempts):
            attempt_seed = master.getrandbits(63)
            rng = random.Random(attempt_seed)
            try:
                floor_map = self._attempt(
                    floor_rule,
                    template,
                    attempt_seed,
                    rng,
                    door_count=door_count,
                    story_count=story_count,
                    evacuation_count=evacuation_count,
                )
                self.validate(floor_map)
                return floor_map
            except MapGenerationError as exc:
                last_error = str(exc)
        raise MapGenerationError(
            f"failed to fill template {template.template_id} after "
            f"{self.config.max_attempts} attempts: {last_error}"
        )

    def _attempt(
        self,
        floor_rule: FloorRule,
        template: MapTemplate,
        seed: int,
        rng: random.Random,
        *,
        door_count: int,
        story_count: int,
        evacuation_count: int,
    ) -> FloorMap:
        occupied = set(template.occupied_slots)
        edges = set(template.edges)
        start = template.start
        distances = _distances(start, edges)
        if set(distances) != occupied:
            raise MapGenerationError(
                f"template {template.template_id} is disconnected or incomplete"
            )
        if max(distances.values()) > floor_rule.maximum_graph_distance:
            raise MapGenerationError(
                f"template {template.template_id} exceeds the floor distance limit"
            )

        assigned: dict[GridPoint, NodeType] = {start: NodeType.START}
        fixed_exits, rule_distances = self._assign_template_terminals(
            floor_rule, template, distances, assigned
        )
        for fixed in template.fixed_slots:
            self._claim(assigned, fixed.slot, fixed.node_type)

        if floor_rule.floor in (1, 2):
            for left, right in edges:
                if left == start:
                    self._claim(assigned, right, NodeType.BATTLE_NORMAL)
                elif right == start:
                    self._claim(assigned, left, NodeType.BATTLE_NORMAL)

        evacuation_exits = self._assign_evacuation(
            floor_rule.floor,
            rule_distances,
            assigned,
            rng,
            evacuation_count,
        )
        exits = fixed_exits + evacuation_exits

        self._assign_content(
            floor_rule.floor,
            rule_distances,
            assigned,
            rng,
            door_count=door_count,
            story_count=story_count,
        )
        if set(assigned) != occupied:
            raise MapGenerationError("content solver left unassigned template slots")

        ordered_points = [start] + sorted(point for point in occupied if point != start)
        id_by_point = {
            point: f"F{floor_rule.floor}_N{index:02d}"
            for index, point in enumerate(ordered_points)
        }
        nodes: list[MapNode] = []
        repeatable_types = {
            NodeType.DOOR,
            NodeType.FINAL,
            NodeType.EVACUATE,
        }
        for index, point in enumerate(ordered_points):
            node_type = assigned[point]
            event_name, options, auto_effect = self._content_for_node(
                floor_rule.floor, node_type, rng
            )
            nodes.append(
                MapNode(
                    node_id=id_by_point[point],
                    index=index,
                    row=point[0],
                    col=point[1],
                    node_type=node_type,
                    distance_from_start=distances[point],
                    options=options,
                    auto_effect=auto_effect,
                    event_name=event_name,
                    repeatable=node_type in repeatable_types,
                )
            )
        mapped_edges = tuple(
            sorted(
                (id_by_point[left], id_by_point[right])
                for left, right in edges
            )
        )
        return FloorMap(
            floor=floor_rule.floor,
            width=floor_rule.width,
            height=floor_rule.height,
            nodes=tuple(nodes),
            edges=mapped_edges,
            start_node_id=id_by_point[start],
            exit_node_ids=tuple(id_by_point[point] for point in exits),
            seed=seed,
        )

    @staticmethod
    def _claim(
        assigned: dict[GridPoint, NodeType],
        point: GridPoint,
        node_type: NodeType,
    ) -> None:
        previous = assigned.get(point)
        if previous is not None and previous is not node_type:
            raise MapGenerationError(
                f"slot {point} is fixed as both {previous.value} and {node_type.value}"
            )
        assigned[point] = node_type

    def _assign_template_terminals(
        self,
        floor_rule: FloorRule,
        template: MapTemplate,
        distances: Mapping[GridPoint, int],
        assigned: dict[GridPoint, NodeType],
    ) -> tuple[tuple[GridPoint, ...], dict[GridPoint, int]]:
        if template.boss_terminal is not None:
            point = template.boss_terminal
            rule = self.ruleset.node_rules[NodeType.BATTLE_BOSS]
            if not rule.allows(floor_rule.floor, distances[point]):
                raise MapGenerationError(
                    f"boss terminal in {template.template_id} has illegal distance"
                )
            self._claim(assigned, point, NodeType.BATTLE_BOSS)
            return (point,), dict(distances)

        terminals = template.final_slots
        final_rule = self.ruleset.node_rules[NodeType.FINAL]
        minimum, maximum = final_rule.count_range(floor_rule.floor)
        if not minimum <= len(terminals) <= maximum:
            raise MapGenerationError(
                f"template {template.template_id} has an invalid FINAL count"
            )
        rule_distances = _rule_distances(
            template.start, template.edges, terminals
        )
        for point in terminals:
            if not final_rule.allows(floor_rule.floor, rule_distances[point]):
                raise MapGenerationError(
                    f"FINAL slot in {template.template_id} has illegal distance"
                )
            self._claim(assigned, point, NodeType.FINAL)
        return terminals, rule_distances

    def _assign_evacuation(
        self,
        floor: int,
        distances: Mapping[GridPoint, int],
        assigned: dict[GridPoint, NodeType],
        rng: random.Random,
        count: int,
    ) -> tuple[GridPoint, ...]:
        rule = self.ruleset.node_rules[NodeType.EVACUATE]
        minimum, maximum = rule.count_range(floor)
        if not minimum <= count <= maximum:
            raise MapGenerationError("evacuation count violates configured range")
        if count == 0:
            return ()
        candidates = [
            point
            for point, distance in distances.items()
            if point not in assigned and rule.allows(floor, distance)
        ]
        if len(candidates) < count:
            raise MapGenerationError("not enough legal slots for EVACUATE")
        selected = tuple(rng.sample(sorted(candidates), k=count))
        for point in selected:
            self._claim(assigned, point, NodeType.EVACUATE)
        return selected

    def _assign_content(
        self,
        floor: int,
        distances: Mapping[GridPoint, int],
        assigned: dict[GridPoint, NodeType],
        rng: random.Random,
        *,
        door_count: int,
        story_count: int,
    ) -> None:
        terminal_types = {
            NodeType.FINAL,
            NodeType.EVACUATE,
            NodeType.BATTLE_BOSS,
        }
        bounds: dict[NodeType, tuple[int, int]] = {}
        active_rules: dict[NodeType, NodeRule] = {}
        for node_type, rule in self.ruleset.node_rules.items():
            if node_type in terminal_types or node_type is NodeType.STORY_HIDDEN:
                continue
            low, high = rule.count_range(floor)
            if node_type is NodeType.DOOR:
                low = high = door_count
            elif node_type is NodeType.STORY:
                low = high = story_count
            elif node_type is NodeType.PORTAL and not self.config.enable_portal:
                low = high = 0
            elif (
                node_type is NodeType.EXPEDITION
                and not self.config.enable_expedition
            ):
                low = high = 0
            elif (
                node_type is NodeType.BATTLE_SAVAGE
                and not self.config.include_advanced_nodes
            ):
                low = high = 0
            bounds[node_type] = (low, high)
            if high > 0:
                active_rules[node_type] = rule

        counts: Counter[NodeType] = Counter(assigned.values())
        for node_type, count in counts.items():
            if node_type in terminal_types or node_type is NodeType.START:
                continue
            _low, high = bounds.get(node_type, (0, 0))
            if count > high:
                raise MapGenerationError(
                    f"fixed {node_type.value} count exceeds configured maximum"
                )
            rule = self.ruleset.node_rules[node_type]
            for point, assigned_type in assigned.items():
                if assigned_type is node_type and not rule.allows(
                    floor, distances[point]
                ):
                    raise MapGenerationError(
                        f"fixed {node_type.value} at illegal distance "
                        f"{distances[point]}"
                    )

        remaining = set(distances) - set(assigned)
        total_ranges = _CONTENT_TOTAL_RANGES[floor]

        def category_of(node_type: NodeType) -> str | None:
            if node_type in _TOTAL_EXCLUDED_TYPES:
                return None
            return self.ruleset.node_rules[node_type].category

        def category_counts() -> Counter[str]:
            result: Counter[str] = Counter()
            for node_type, count in counts.items():
                category = category_of(node_type)
                if category is not None:
                    result[category] += count
            return result

        minimums: dict[NodeType, int] = {}
        capacities: dict[NodeType, int] = {}
        for node_type, (low, high) in bounds.items():
            rule = active_rules.get(node_type)
            capacity = counts[node_type]
            if rule is not None:
                capacity += sum(
                    rule.allows(floor, distances[point]) for point in remaining
                )
            minimums[node_type] = max(low, counts[node_type])
            capacities[node_type] = min(high, capacity)
            if minimums[node_type] > capacities[node_type]:
                raise MapGenerationError(
                    f"not enough legal slots for {node_type.value}"
                )

        content_slot_count = len(remaining) + sum(
            counts[node_type] for node_type in bounds
        )
        empty_min = minimums[NodeType.EMPTY]
        empty_max = capacities[NodeType.EMPTY]
        category_minimums = {
            category: sum(
                minimums[node_type]
                for node_type in bounds
                if category_of(node_type) == category
            )
            for category in total_ranges
        }
        category_capacities = {
            category: sum(
                capacities[node_type]
                for node_type in bounds
                if category_of(node_type) == category
            )
            for category in total_ranges
        }
        feasible_totals: list[tuple[int, int, int]] = []
        mystery_range = total_ranges["mystery"]
        ferocity_range = total_ranges["ferocity"]
        for empty_target in range(empty_min, empty_max + 1):
            for mystery_target in range(
                max(mystery_range[0], category_minimums["mystery"]),
                min(mystery_range[1], category_capacities["mystery"]) + 1,
            ):
                ferocity_target = content_slot_count - empty_target - mystery_target
                if (
                    max(ferocity_range[0], category_minimums["ferocity"])
                    <= ferocity_target
                    <= min(
                        ferocity_range[1],
                        category_capacities["ferocity"],
                    )
                ):
                    feasible_totals.append(
                        (empty_target, mystery_target, ferocity_target)
                    )
        if not feasible_totals:
            raise MapGenerationError(
                "no feasible EMPTY/mystery/ferocity total for template"
            )
        empty_target, mystery_target, ferocity_target = rng.choice(feasible_totals)

        targets = dict(minimums)
        targets[NodeType.EMPTY] = empty_target
        for category, category_target in (
            ("mystery", mystery_target),
            ("ferocity", ferocity_target),
        ):
            needed = category_target - sum(
                targets[node_type]
                for node_type in bounds
                if category_of(node_type) == category
            )
            while needed > 0:
                options = [
                    node_type
                    for node_type in bounds
                    if category_of(node_type) == category
                    and targets[node_type] < capacities[node_type]
                ]
                if not options:
                    raise MapGenerationError(
                        f"cannot allocate the {category} content quota"
                    )
                weights = [
                    self.ruleset.node_rules[node_type].weight
                    if self.ruleset.node_rules[node_type].weight > 0
                    else _SPECIAL_FALLBACK_WEIGHTS.get(node_type, 0.1)
                    for node_type in options
                ]
                selected = rng.choices(options, weights=weights, k=1)[0]
                targets[selected] += 1
                needed -= 1
        bounds = {
            node_type: (targets[node_type], targets[node_type])
            for node_type in bounds
        }

        def candidates_for(point: GridPoint) -> list[NodeRule]:
            result: list[NodeRule] = []
            current_categories = category_counts()
            for node_type, rule in active_rules.items():
                _low, high = bounds[node_type]
                if counts[node_type] >= high:
                    continue
                if not rule.allows(floor, distances[point]):
                    continue
                category = category_of(node_type)
                if (
                    category is not None
                    and current_categories[category] >= total_ranges[category][1]
                ):
                    continue
                result.append(rule)
            return result

        def forward_feasible() -> bool:
            if sum(
                max(0, low - counts[node_type])
                for node_type, (low, _high) in bounds.items()
            ) > len(remaining):
                return False

            for node_type, (low, high) in bounds.items():
                if counts[node_type] > high:
                    return False
                if counts[node_type] >= low:
                    continue
                rule = active_rules.get(node_type)
                if rule is None:
                    return False
                possible = sum(
                    rule.allows(floor, distances[point]) for point in remaining
                )
                if counts[node_type] + possible < low:
                    return False

            current_categories = category_counts()
            for category, (low, high) in total_ranges.items():
                if current_categories[category] > high:
                    return False
                if current_categories[category] >= low:
                    continue
                possible = sum(
                    any(
                        category_of(rule.node_type) == category
                        for rule in candidates_for(point)
                    )
                    for point in remaining
                )
                if current_categories[category] + possible < low:
                    return False

            return all(candidates_for(point) for point in remaining)

        def weighted_order(options: list[NodeRule]) -> list[NodeRule]:
            pool = list(options)
            ordered: list[NodeRule] = []
            while pool:
                weights = [
                    rule.weight
                    if rule.weight > 0
                    else _SPECIAL_FALLBACK_WEIGHTS.get(rule.node_type, 0.1)
                    for rule in pool
                ]
                selected = rng.choices(pool, weights=weights, k=1)[0]
                pool.remove(selected)
                ordered.append(selected)
            return ordered

        search_steps = 0

        def solve() -> bool:
            nonlocal search_steps
            search_steps += 1
            if search_steps > 250_000 or not forward_feasible():
                return False
            if not remaining:
                current_categories = category_counts()
                return all(
                    low <= counts[node_type] <= high
                    for node_type, (low, high) in bounds.items()
                ) and all(
                    low <= current_categories[category] <= high
                    for category, (low, high) in total_ranges.items()
                )

            option_sets = [
                (len(options), point, options)
                for point in remaining
                if (options := candidates_for(point))
            ]
            if len(option_sets) != len(remaining):
                return False
            minimum = min(item[0] for item in option_sets)
            tied = [item for item in option_sets if item[0] == minimum]
            _size, point, options = rng.choice(tied)
            remaining.remove(point)
            for rule in weighted_order(options):
                assigned[point] = rule.node_type
                counts[rule.node_type] += 1
                if solve():
                    return True
                counts[rule.node_type] -= 1
                del assigned[point]
            remaining.add(point)
            return False

        if not solve():
            raise MapGenerationError(
                "node-content constraints are unsatisfiable for this template"
            )

    def _content_for_node(
        self,
        floor: int,
        node_type: NodeType,
        rng: random.Random,
    ) -> tuple[str | None, tuple[EventOption, ...], ResourceDelta]:
        if node_type is NodeType.BATTLE_NORMAL:
            return None, (), ResourceDelta(
                gold=rng.randint(3, 6),
                hope=rng.randint(0, 1),
                parts=int(rng.random() < 0.35),
                tickets=int(rng.random() < 0.25),
                team_strength=1,
            )
        if node_type is NodeType.BATTLE_ELITE:
            return None, (), ResourceDelta(
                gold=rng.randint(6, 10), hope=1, parts=1, relics=1, team_strength=1
            )
        if node_type is NodeType.BATTLE_BOSS:
            return None, (), ResourceDelta(gold=10, hope=3, parts=2, relics=2, team_strength=2)
        if node_type is NodeType.BATTLE_SAVAGE:
            return None, (), ResourceDelta(gold=8, parts=2, team_strength=1)
        if node_type is NodeType.EMPTY or node_type is NodeType.START:
            return None, (), ResourceDelta()
        if node_type is NodeType.LIGHT:
            return None, (), ResourceDelta(action_points=1)
        if node_type is NodeType.FINAL:
            return None, (), ResourceDelta(parts=1)
        if node_type is NodeType.EVACUATE:
            return None, (), ResourceDelta(parts=1)

        leave = EventOption("leave", "离开", description="不改变当前资源")
        if node_type is NodeType.BATTLE_SHOP:
            return "诡意行商", (
                EventOption("buy_relic", "6锭购买收藏品", ResourceDelta(gold=-6, relics=1)),
                EventOption("buy_ticket", "4锭购买招募券", ResourceDelta(gold=-4, tickets=1)),
                EventOption("buy_beta", "10锭购买沙盘β", ResourceDelta(gold=-10), add_items=("beta",)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.SCRAP_SHOP:
            return "秘境行商", (
                EventOption("buy_parts", "5锭采购2件零件", ResourceDelta(gold=-5, parts=2)),
                EventOption("sell_parts", "出售1件零件换8锭", ResourceDelta(gold=8, parts=-1)),
                EventOption("cultivate", "培育自然物", ResourceDelta(parts=-1, relics=1, hope=1)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.REST:
            options = [
                EventOption("rest_hp", "生命上限+3", ResourceDelta(max_hp=3)),
                EventOption("rest_shield", "获得5护盾", ResourceDelta(shield=5)),
                EventOption("rest_hope", "获得3希望", ResourceDelta(hope=3)),
                EventOption("rest_ap", "恢复2行动力", ResourceDelta(action_points=2)),
                EventOption("rest_part", "扩充零件储备", ResourceDelta(parts=1)),
            ]
            rng.shuffle(options)
            return "安全的角落", tuple(options[:3]), ResourceDelta()
        if node_type is NodeType.INCIDENT:
            event_name = rng.choice(self.ruleset.event_pools[floor])
            return event_name, self._incident_options(event_name, rng), ResourceDelta()
        if node_type is NodeType.WISH:
            return "得偿所愿", (
                EventOption("wish_relic", "选择免费收藏品", ResourceDelta(relics=1)),
                EventOption("wish_refresh", "花4锭刷新并取两件", ResourceDelta(gold=-4, relics=2)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.SACRIFICE:
            return "失与得", (
                EventOption("trade_relic", "交换1件收藏品", ResourceDelta(relics=-1, parts=2, hope=2)),
                EventOption("trade_parts", "交换2件零件", ResourceDelta(parts=-2, relics=1, gold=5)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.EXPEDITION:
            return "先行一步", (
                EventOption(
                    "seek_source",
                    "派遣干员探索源头",
                    ResourceDelta(team_strength=-1, hope=4),
                    add_items=("beacon",),
                ),
                EventOption("decline", "不派遣，获得2希望", ResourceDelta(hope=2)),
            ), ResourceDelta()
        if node_type is NodeType.PORTAL:
            return "误入奇境", (
                EventOption("enter_portal", "消耗加工品进入黑潭", ResourceDelta(parts=-1, action_points=3, relics=1)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.DUEL:
            return "狭路相逢", (
                EventOption("duel_parts", "搏杀：争取高稀有零件", ResourceDelta(parts=2, tickets=2), battle=True),
                EventOption("duel_mixed", "共斗：零件与收藏品", ResourceDelta(parts=1, relics=1, tickets=2), battle=True),
                EventOption("duel_relic", "共斗：争取收藏品", ResourceDelta(relics=2, tickets=2), battle=True),
            ), ResourceDelta()
        if node_type is NodeType.STORY and floor == 6:
            return "调谐仪式", (
                EventOption("tune", "完成调谐仪式"),
            ), ResourceDelta()
        if node_type in {NodeType.STORY, NodeType.STORY_HIDDEN}:
            return "命运所指", (
                EventOption("protect", "保护证据", ResourceDelta(hope=5), add_items=("ending_2",)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.EMPLOY:
            return "应急助力", (
                EventOption("free_hire", "雇佣临时干员", ResourceDelta(team_strength=1)),
                EventOption("paid_hire", "4锭雇佣精英干员", ResourceDelta(gold=-4, team_strength=2)),
                leave,
            ), ResourceDelta()
        if node_type is NodeType.DOOR:
            return "曲折密道", (), ResourceDelta()
        return self.ruleset.node_rules[node_type].label, (leave,), ResourceDelta()

    def _incident_options(
        self, event_name: str, rng: random.Random
    ) -> tuple[EventOption, ...]:
        leave = EventOption("leave", "离开")
        if event_name == "桑尼的邀请":
            return (
                EventOption("hope", "获得3希望", ResourceDelta(hope=3)),
                EventOption("gold", "获得10源石锭", ResourceDelta(gold=10)),
            )
        if event_name == "色味不同源":
            return (
                EventOption("max_hp", "生命上限+3", ResourceDelta(max_hp=3)),
                EventOption("shield", "获得5护盾", ResourceDelta(shield=5)),
            )
        if event_name == "敲动杠杆":
            return (
                EventOption("small_trade", "4锭换1件收藏品", ResourceDelta(gold=-4, relics=1)),
                EventOption("large_trade", "10锭换3件收藏品", ResourceDelta(gold=-10, relics=3)),
                leave,
            )
        if event_name == "沉重的契约":
            return (
                EventOption("safe_ap", "获得2行动力", ResourceDelta(action_points=2)),
                EventOption("risky_ap", "失去3生命，获得5行动力", ResourceDelta(hp=-3, action_points=5)),
                leave,
            )
        if event_name == "沉寂之屋":
            return (
                EventOption("hp_cage", "失去2生命取得笼控器", ResourceDelta(hp=-2), add_items=("cage",)),
                EventOption("gold_cage", "支付8锭取得笼控器", ResourceDelta(gold=-8), add_items=("cage",)),
                leave,
            )
        if event_name == "线人":
            return (
                EventOption("alpha", "取得沙盘α", ResourceDelta(hope=1), add_items=("alpha",)),
                leave,
            )
        if event_name == "泪之聚落":
            return (
                EventOption("pay_gold", "支付12锭取得关键线索", ResourceDelta(gold=-12), add_items=("ending_3_key",)),
                leave,
            )
        if event_name == "擒与缚":
            return (
                EventOption("parts", "取得翱翼与虬蜕", ResourceDelta(parts=2)),
                EventOption("relic", "取得收藏品", ResourceDelta(relics=1)),
            )
        if event_name in {"黑诞", "湖中仙女", "鸭托邦", "传奇团伙"}:
            return (
                EventOption("battle", "接受特殊作战", ResourceDelta(gold=8, relics=1, parts=1), battle=True),
                leave,
            )
        # The public data has prose but no machine-readable effect linkage.
        # These two deterministic choices are the explicitly versioned prior,
        # not a claim about the exact client event.
        bonus = rng.randint(6, 10)
        return (
            EventOption("resource", "取得补给", ResourceDelta(gold=bonus, hope=1)),
            EventOption("risk", "迎战并夺取稀有物资", ResourceDelta(parts=1, relics=1), battle=True),
            leave,
        )

    def validate(self, floor_map: FloorMap) -> None:
        floor_rule = self.ruleset.floor(floor_map.floor)
        if (floor_map.width, floor_map.height) != (
            floor_rule.width,
            floor_rule.height,
        ):
            raise MapGenerationError("map dimensions do not match floor rules")
        low_nodes, high_nodes = floor_rule.node_count
        if not low_nodes <= len(floor_map.nodes) <= high_nodes:
            raise MapGenerationError("node count violates floor range")
        if len(floor_map.nodes) > self.ruleset.max_nodes:
            raise MapGenerationError("node count exceeds policy padding")
        ids = {node.node_id for node in floor_map.nodes}
        if len(ids) != len(floor_map.nodes) or floor_map.start_node_id not in ids:
            raise MapGenerationError("node ids are invalid")
        if tuple(node.index for node in floor_map.nodes) != tuple(range(len(floor_map.nodes))):
            raise MapGenerationError("node indices are not contiguous")
        positions = {(node.row, node.col): node for node in floor_map.nodes}
        if len(positions) != len(floor_map.nodes):
            raise MapGenerationError("duplicate grid cell")
        if any(
            not (0 <= node.row < floor_map.height and 0 <= node.col < floor_map.width)
            for node in floor_map.nodes
        ):
            raise MapGenerationError("node coordinate is outside the grid")

        starts = [node for node in floor_map.nodes if node.node_type is NodeType.START]
        if (
            len(starts) != 1
            or starts[0].node_id != floor_map.start_node_id
            or starts[0].distance_from_start != 0
        ):
            raise MapGenerationError("map must have exactly one valid start node")

        seen_edges: set[tuple[str, str]] = set()
        coordinate_edges: set[tuple[GridPoint, GridPoint]] = set()
        for left, right in floor_map.edges:
            if left not in ids or right not in ids:
                raise MapGenerationError("edge references unknown node")
            normalized = tuple(sorted((left, right)))
            if left == right or normalized in seen_edges:
                raise MapGenerationError("map contains a duplicate or self edge")
            seen_edges.add(normalized)
            a, b = floor_map.node(left), floor_map.node(right)
            if abs(a.row - b.row) + abs(a.col - b.col) != 1:
                raise MapGenerationError("edge is not four-directional")
            coordinate_edges.add(
                _ordered_edge((a.row, a.col), (b.row, b.col))
            )

        adjacency = floor_map.adjacency()
        seen = {floor_map.start_node_id}
        computed_distance = {floor_map.start_node_id: 0}
        queue = deque([floor_map.start_node_id])
        while queue:
            node_id = queue.popleft()
            for other in adjacency[node_id]:
                if other not in seen:
                    seen.add(other)
                    computed_distance[other] = computed_distance[node_id] + 1
                    queue.append(other)
        if seen != ids:
            raise MapGenerationError("map is disconnected")

        for node in floor_map.nodes:
            if node.distance_from_start != computed_distance[node.node_id]:
                raise MapGenerationError("stored node distance does not match graph BFS")
        if max(computed_distance.values()) > floor_rule.maximum_graph_distance:
            raise MapGenerationError("graph distance exceeds floor maximum")

        listed_exits = set(floor_map.exit_node_ids)
        actual_exits = {node.node_id for node in floor_map.nodes if node.is_exit}
        if (
            not listed_exits
            or len(listed_exits) != len(floor_map.exit_node_ids)
            or listed_exits != actual_exits
        ):
            raise MapGenerationError("exit_node_ids must exactly list every exit")
        exit_nodes = [floor_map.node(node_id) for node_id in listed_exits]
        rule_distances = _rule_distances(
            (starts[0].row, starts[0].col),
            coordinate_edges,
            {
                (node.row, node.col)
                for node in floor_map.nodes
                if node.node_type is NodeType.FINAL
            },
        )
        if any(
            node.node_type not in floor_rule.exit_types
            or not self.ruleset.node_rules[node.node_type].allows(
                floor_map.floor, rule_distances[(node.row, node.col)]
            )
            for node in exit_nodes
        ):
            raise MapGenerationError("invalid exit type or distance")

        template = self.templates.find_match(floor_map)
        if template is None:
            raise MapGenerationError("map topology is not one of the verified templates")
        for point, node_type in template.fixed_slot_types.items():
            if positions[point].node_type is not node_type:
                raise MapGenerationError(
                    f"template fixed slot {point} must be {node_type.value}"
                )
        if template.boss_terminal is not None:
            if positions[template.boss_terminal].node_type is not NodeType.BATTLE_BOSS:
                raise MapGenerationError("boss template terminal must be BATTLE_BOSS")
        elif any(
            positions[point].node_type is not NodeType.FINAL
            for point in template.final_slots
        ):
            raise MapGenerationError("template FINAL slots must remain FINAL")
        actual_final_slots = {
            (node.row, node.col)
            for node in floor_map.nodes
            if node.node_type is NodeType.FINAL
        }
        if actual_final_slots != set(template.final_slots):
            raise MapGenerationError("FINAL nodes must exactly match template FINAL slots")
        if floor_map.floor in (1, 2):
            for node_id in adjacency[floor_map.start_node_id]:
                if floor_map.node(node_id).node_type is not NodeType.BATTLE_NORMAL:
                    raise MapGenerationError(
                        "every start neighbour on floor I/II must be BATTLE_NORMAL"
                    )

        type_counts = Counter(node.node_type for node in floor_map.nodes)
        if type_counts[NodeType.DOOR] not in {0, 2}:
            raise MapGenerationError("DOOR count must be exactly 0 or 2")
        expected_story = (
            1
            if floor_map.floor == 6
            else 3
            if floor_map.floor == 5 and self.config.enable_second_ending
            else 0
        )
        if type_counts[NodeType.STORY] != expected_story:
            raise MapGenerationError(
                f"STORY count must be {expected_story} for the configured ending route"
            )
        gated_types = {
            NodeType.PORTAL: self.config.enable_portal,
            NodeType.EXPEDITION: self.config.enable_expedition,
            NodeType.BATTLE_SAVAGE: self.config.include_advanced_nodes,
        }
        for node_type, enabled in gated_types.items():
            if not enabled and type_counts[node_type]:
                raise MapGenerationError(
                    f"{node_type.value} requires its generation gate"
                )

        content_totals: Counter[str] = Counter()
        for node_type, count in type_counts.items():
            if node_type in _TOTAL_EXCLUDED_TYPES:
                continue
            content_totals[self.ruleset.node_rules[node_type].category] += count
        for category, (minimum, maximum) in _CONTENT_TOTAL_RANGES[
            floor_map.floor
        ].items():
            if not minimum <= content_totals[category] <= maximum:
                raise MapGenerationError(
                    f"{category} total violates configured range"
                )

        for node_type, rule in self.ruleset.node_rules.items():
            minimum, maximum = rule.count_range(floor_map.floor)
            if not minimum <= type_counts[node_type] <= maximum:
                raise MapGenerationError(
                    f"{node_type.value} count violates configured range"
                )
        for node in floor_map.nodes:
            if node.node_type is NodeType.START:
                continue
            rule = self.ruleset.node_rules[node.node_type]
            placement_distance = rule_distances[(node.row, node.col)]
            if not rule.allows(floor_map.floor, placement_distance):
                raise MapGenerationError(
                    f"{node.node_type.value} at illegal rule distance "
                    f"{placement_distance}"
                )


def _grid_neighbours(point: GridPoint, width: int, height: int) -> tuple[GridPoint, ...]:
    row, col = point
    return tuple(
        (other_row, other_col)
        for other_row, other_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        if 0 <= other_row < height and 0 <= other_col < width
    )


def _ordered_edge(left: GridPoint, right: GridPoint) -> tuple[GridPoint, GridPoint]:
    return (left, right) if left < right else (right, left)


def _distances(
    start: GridPoint, edges: Iterable[tuple[GridPoint, GridPoint]]
) -> dict[GridPoint, int]:
    adjacency: dict[GridPoint, list[GridPoint]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    result = {start: 0}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for other in adjacency.get(point, []):
            if other not in result:
                result[other] = result[point] + 1
                queue.append(other)
    return result


def _rule_distances(
    start: GridPoint,
    edges: Iterable[tuple[GridPoint, GridPoint]],
    final_points: Iterable[GridPoint],
) -> dict[GridPoint, int]:
    """Compute the route-tool placement metric without changing physical edges.

    For prediction only, every pair of physical neighbours around the same
    FINAL terminal is connected directly.  This makes crossing that terminal
    cost one placement-distance step instead of two; runtime movement continues
    to use the original template graph.
    """

    physical_edges = {_ordered_edge(left, right) for left, right in edges}
    adjacency: dict[GridPoint, set[GridPoint]] = {}
    for left, right in physical_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    rule_edges = set(physical_edges)
    for final_point in final_points:
        neighbours = sorted(adjacency.get(final_point, ()))
        for index, left in enumerate(neighbours):
            for right in neighbours[index + 1 :]:
                rule_edges.add(_ordered_edge(left, right))
    return _distances(start, rule_edges)
