from __future__ import annotations

from collections import deque
from dataclasses import replace
from hashlib import sha256
from typing import Iterable

from .domain import (
    Action,
    ActionKind,
    BATTLE_TYPES,
    EventOption,
    FloorMap,
    GameState,
    MapNode,
    NodeType,
    ResourceDelta,
    ResourceState,
    Transition,
)
from .mapgen import MapGenerator
from .rules import Ruleset, load_ruleset


class InvalidAction(ValueError):
    pass


class BlackflowSimulator:
    """Pure-transition environment for route and event decisions.

    Combat is resolved as a guaranteed victory, as requested.  Random map and
    reward draws are fixed at ``reset(seed)``; ``transition`` itself is fully
    deterministic, which prevents MCTS from cloning or peeking at mutable RNG
    state.
    """

    def __init__(
        self,
        ruleset: Ruleset | None = None,
        map_generator: MapGenerator | None = None,
    ) -> None:
        self.ruleset = ruleset or load_ruleset()
        self.map_generator = map_generator or MapGenerator(self.ruleset)
        if self.map_generator.ruleset.sha256 != self.ruleset.sha256:
            raise ValueError("map generator and simulator rulesets differ")
        self._future_belief_maps: dict[int, FloorMap] = {}

    @property
    def action_size(self) -> int:
        return self.ruleset.action_size

    def reset(
        self,
        seed: int,
        *,
        resources: ResourceState | None = None,
        inventory: Iterable[str] = (),
    ) -> GameState:
        maps = self.map_generator.generate_run(seed)
        first = maps[0]
        initial = resources or ResourceState()
        initial = replace(
            initial,
            action_points=self.ruleset.floor(first.floor).action_points,
        )
        state = GameState(
            maps=maps,
            floor_index=0,
            current_node_id=first.start_node_id,
            resources=initial,
            completed=frozenset({first.start_node_id}),
            revealed=frozenset({first.start_node_id}),
            inventory=frozenset(inventory),
            history=(f"进入第{first.floor}层（seed={seed}）",),
        )
        return self._refresh_revealed(state)

    def legal_action_ids(self, state: GameState) -> tuple[int, ...]:
        return tuple(action.action_id for action in self.legal_actions(state))

    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        if state.terminal:
            return ()
        if state.pending_node_id is not None:
            node = state.floor_map.node(state.pending_node_id)
            result = []
            for option_index, option in enumerate(node.options[: self.ruleset.max_options]):
                if option.is_available(state.resources, state.inventory):
                    result.append(
                        Action(
                            action_id=self.ruleset.max_nodes + option_index,
                            kind=ActionKind.CHOOSE,
                            target_node_id=node.node_id,
                            option_index=option_index,
                        )
                    )
            return tuple(result)

        frontier = self.reachable_frontier(state)
        result = []
        for node_id, movement_cost in sorted(
            frontier.items(), key=lambda item: state.floor_map.node(item[0]).index
        ):
            if movement_cost <= state.resources.action_points:
                node = state.floor_map.node(node_id)
                result.append(
                    Action(
                        action_id=node.index,
                        kind=ActionKind.MOVE,
                        target_node_id=node_id,
                        movement_cost=movement_cost,
                    )
                )
        return tuple(result)

    def decode_action(self, state: GameState, action_id: int) -> Action:
        for action in self.legal_actions(state):
            if action.action_id == action_id:
                return action
        raise InvalidAction(
            f"action {action_id} is illegal; legal={self.legal_action_ids(state)}"
        )

    def transition(self, state: GameState, action_id: int) -> Transition:
        action = self.decode_action(state, action_id)
        messages: list[str] = []
        reward = 0.0
        next_state = state

        if action.kind is ActionKind.MOVE:
            assert action.target_node_id is not None
            node = state.floor_map.node(action.target_node_id)
            before = state.resources
            after = before.apply(ResourceDelta(action_points=-action.movement_cost))
            reward += self.ruleset.objective.resource_reward(before, after)
            next_state = replace(
                state,
                current_node_id=node.node_id,
                resources=after,
                revealed=state.revealed | {node.node_id},
            )
            messages.append(
                f"移动到{self.node_label(node)}，消耗{action.movement_cost}行动力"
            )
            if node.options:
                next_state = replace(next_state, pending_node_id=node.node_id)
            else:
                next_state, gained = self._complete_node(next_state, node, node.auto_effect)
                reward += gained
                if node.is_exit:
                    next_state, gained, exit_messages = self._advance_floor(next_state, node)
                    reward += gained
                    messages.extend(exit_messages)
                else:
                    next_state = self._refresh_revealed(next_state)
                    if self._must_trigger_chase(next_state):
                        next_state, gained, chase_messages = self._trigger_chase(next_state)
                        reward += gained
                        messages.extend(chase_messages)
        else:
            assert action.option_index is not None
            assert state.pending_node_id is not None
            node = state.floor_map.node(state.pending_node_id)
            option = node.options[action.option_index]
            if not option.is_available(state.resources, state.inventory):
                raise InvalidAction(f"option {option.option_id} no longer meets its requirements")
            effect = option.effect
            if option.battle:
                # Any combat embedded in an event is also a guaranteed victory.
                effect = effect + ResourceDelta(gold=4, tickets=1, team_strength=1)
            new_inventory = (state.inventory - set(option.remove_items)) | set(option.add_items)
            before = state.resources
            after = before.apply(effect)
            key_items_added = len(set(option.add_items) - set(state.inventory))
            reward += self.ruleset.objective.resource_reward(
                before, after, key_items_added=key_items_added
            )
            next_state = replace(
                state,
                resources=after,
                inventory=frozenset(new_inventory),
                pending_node_id=None,
                completed=state.completed | {node.node_id},
            )
            messages.append(f"在{self.node_label(node)}选择“{option.title}”")
            next_state = self._refresh_revealed(next_state)
            if self._must_trigger_chase(next_state):
                next_state, gained, chase_messages = self._trigger_chase(next_state)
                reward += gained
                messages.extend(chase_messages)

        message = "；".join(messages)
        next_state = next_state.with_reward(reward, message)
        return Transition(
            next_state=next_state,
            reward=reward,
            terminated=next_state.terminal,
            info={"action": action, "messages": tuple(messages)},
        )

    def reachable_frontier(self, state: GameState) -> dict[str, int]:
        """Return incomplete nodes reachable through completed nodes and their AP cost."""

        adjacency = state.floor_map.adjacency()
        completed = state.completed
        doors = [
            node.node_id
            for node in state.floor_map.nodes
            if node.node_type is NodeType.DOOR
        ]
        door_pair = (
            {doors[0]: doors[1], doors[1]: doors[0]}
            if len(doors) == 2
            else {}
        )
        distances = {state.current_node_id: 0}
        queue = deque([state.current_node_id])
        frontier: dict[str, int] = {}
        while queue:
            node_id = queue.popleft()
            base = distances[node_id]
            neighbours = [(other, 1) for other in adjacency[node_id]]
            if node_id in door_pair:
                neighbours.append((door_pair[node_id], 0))
            for other, step_cost in neighbours:
                cost = base + step_cost
                if other in completed:
                    if cost < distances.get(other, 10**9):
                        distances[other] = cost
                        if step_cost == 0:
                            queue.appendleft(other)
                        else:
                            queue.append(other)
                else:
                    frontier[other] = min(frontier.get(other, cost), cost)
        return frontier

    def action_description(self, state: GameState, action_id: int) -> str:
        action = self.decode_action(state, action_id)
        if action.kind is ActionKind.MOVE:
            node = state.floor_map.node(action.target_node_id or "")
            visibility = self.node_label(node) if node.node_id in state.revealed else self.hidden_label(node)
            return f"前往 {visibility} ({node.node_id}, {action.movement_cost}行动力)"
        node = state.floor_map.node(state.pending_node_id or "")
        option = node.options[action.option_index or 0]
        return f"选择 {option.title}"

    def node_label(self, node: MapNode) -> str:
        if node.node_type is NodeType.START:
            return "起点"
        return self.ruleset.node_rules[node.node_type].label

    def hidden_label(self, node: MapNode) -> str:
        return "未知的凶戾" if node.node_type in BATTLE_TYPES else "未知的诡秘"

    def render_text(self, state: GameState) -> str:
        floor_map = state.floor_map
        cells = {(node.row, node.col): node for node in floor_map.nodes}
        rows: list[str] = []
        for row in range(floor_map.height):
            rendered: list[str] = []
            for col in range(floor_map.width):
                node = cells.get((row, col))
                if node is None:
                    rendered.append("   ·   ")
                    continue
                if node.node_id == state.current_node_id:
                    marker = "@"
                elif node.node_id in state.completed:
                    marker = "✓"
                elif node.node_id in state.revealed:
                    marker = "?"
                else:
                    marker = "·"
                if node.node_id in state.revealed:
                    label = self.node_label(node)[:4]
                else:
                    label = "凶戾" if node.is_battle else "诡秘"
                rendered.append(f"{marker}{label:<4}")
            rows.append(" ".join(rendered))
        r = state.resources
        header = (
            f"第{state.floor}层  AP={r.action_points}  HP={r.hp}/{r.max_hp}  "
            f"锭={r.gold} 希望={r.hope} 零件={r.parts} 收藏品={r.relics}"
        )
        return header + "\n" + "\n".join(rows)

    def belief_state(self, state: GameState) -> GameState:
        """Build a deterministic expected-world state for leakage-free MCTS.

        The real hidden node is kept in the episode state, but search receives
        a representative outcome based only on the observation category.  This
        is an MVP belief approximation: once real event probabilities are
        available it should be replaced by root belief sampling/chance nodes.
        """

        expected_event_options = (
            EventOption("expected_supply", "预期补给", ResourceDelta(gold=6, hope=1)),
            EventOption(
                "expected_battle",
                "预期特殊作战",
                ResourceDelta(parts=1, relics=1),
                battle=True,
            ),
            EventOption("expected_leave", "离开"),
        )
        maps = []
        for map_index, real_floor_map in enumerate(state.maps):
            # A player has not observed later floors at all.  Searching their
            # true topology or a hash derived from their hidden node types is
            # therefore a form of look-ahead leakage.  Replace each future
            # floor with a ruleset-only, canonical determinization.
            floor_map = (
                real_floor_map
                if map_index <= state.floor_index
                else self._future_belief_map(real_floor_map.floor)
            )
            nodes = []
            for node in floor_map.nodes:
                # Exact payload becomes observable only after completion or
                # while the event-choice UI is open.  Seeing a node's type on
                # the map must not reveal its rolled event, options, or loot.
                if (
                    node.node_type is NodeType.START
                    or node.node_id in state.completed
                    or node.node_id == state.pending_node_id
                ):
                    nodes.append(node)
                    continue
                if node.node_id in state.revealed or node.is_exit:
                    observed_type = node.node_type
                else:
                    observed_type = (
                        NodeType.BATTLE_NORMAL
                        if node.is_battle
                        else NodeType.INCIDENT
                    )
                nodes.append(
                    self._expected_unentered_node(
                        node, observed_type, expected_event_options
                    )
                )
            # `FloorMap.seed` and the original fingerprint are generator
            # internals, not player observations.  Reset both so an alternate
            # evaluator cannot reverse them into hidden contents.
            maps.append(
                replace(
                    floor_map,
                    nodes=tuple(nodes),
                    seed=0,
                    fingerprint="",
                )
            )
        return replace(state, maps=tuple(maps), history=())

    def _future_belief_map(self, floor: int) -> FloorMap:
        cached = self._future_belief_maps.get(floor)
        if cached is not None:
            return cached
        digest = sha256(
            f"{self.ruleset.ruleset_id}:{self.ruleset.sha256}:future-belief:{floor}".encode()
        ).digest()
        # Keep the seed in the positive signed 63-bit range used elsewhere.
        seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        generated = self.map_generator.generate_floor(floor, seed)
        self._future_belief_maps[floor] = generated
        return generated

    def _expected_unentered_node(
        self,
        node: MapNode,
        node_type: NodeType,
        event_options: tuple[EventOption, ...],
    ) -> MapNode:
        battle_rewards = {
            NodeType.BATTLE_NORMAL: ResourceDelta(gold=5, team_strength=1),
            NodeType.BATTLE_ELITE: ResourceDelta(
                gold=8, hope=1, parts=1, relics=1, team_strength=1
            ),
            NodeType.BATTLE_BOSS: ResourceDelta(
                gold=10, hope=3, parts=2, relics=2, team_strength=2
            ),
            NodeType.BATTLE_SAVAGE: ResourceDelta(
                gold=8, parts=2, team_strength=1
            ),
        }
        if node_type in battle_rewards:
            return replace(
                node,
                node_type=node_type,
                options=(),
                auto_effect=battle_rewards[node_type],
                event_name=None,
            )
        fixed_effects = {
            NodeType.START: ResourceDelta(),
            NodeType.EMPTY: ResourceDelta(),
            NodeType.LIGHT: ResourceDelta(action_points=1),
            NodeType.DOOR: ResourceDelta(),
            NodeType.FINAL: ResourceDelta(parts=1),
            NodeType.EVACUATE: ResourceDelta(parts=1),
        }
        if node_type in fixed_effects:
            return replace(
                node,
                node_type=node_type,
                options=(),
                auto_effect=fixed_effects[node_type],
                event_name=None,
            )
        return replace(
            node,
            node_type=node_type,
            options=event_options,
            auto_effect=ResourceDelta(),
            event_name="未进入节点期望模型",
        )

    def _complete_node(
        self,
        state: GameState,
        node: MapNode,
        effect: ResourceDelta,
    ) -> tuple[GameState, float]:
        before = state.resources
        after = before.apply(effect)
        reward = self.ruleset.objective.resource_reward(before, after)
        return (
            replace(
                state,
                resources=after,
                completed=state.completed | {node.node_id},
                pending_node_id=None,
            ),
            reward,
        )

    def _advance_floor(
        self,
        state: GameState,
        exit_node: MapNode,
        *,
        chased: bool = False,
    ) -> tuple[GameState, float, list[str]]:
        reward = self.ruleset.objective.floor_clear_bonus
        messages = [f"完成第{state.floor}层"]
        resources = state.resources
        carry = 0
        if not chased and exit_node.node_type is NodeType.EVACUATE:
            carry = resources.action_points
            resources = replace(resources, action_points=0)
            messages.append(f"险路小径保留{carry}行动力")
        elif not chased:
            remaining = resources.action_points
            before = resources
            resources = resources.apply(
                ResourceDelta(hope=remaining, action_points=-remaining)
            )
            reward += self.ruleset.objective.resource_reward(before, resources)
            if remaining:
                messages.append(f"剩余{remaining}行动力转化为希望")

        if state.floor_index + 1 >= len(state.maps):
            reward += self.ruleset.objective.run_clear_bonus
            messages.append("完成整次探索")
            return replace(state, resources=resources, terminal=True), reward, messages

        next_index = state.floor_index + 1
        next_map = state.maps[next_index]
        base_ap = self.ruleset.floor(next_map.floor).action_points
        if chased:
            base_ap = max(1, base_ap - 1)
            messages.append("追猎导致下层初始行动力-1")
        resources = replace(resources, action_points=base_ap + carry)
        next_state = replace(
            state,
            floor_index=next_index,
            current_node_id=next_map.start_node_id,
            resources=resources,
            completed=state.completed | {next_map.start_node_id},
            revealed=state.revealed | {next_map.start_node_id},
            pending_node_id=None,
        )
        messages.append(f"进入第{next_map.floor}层")
        return self._refresh_revealed(next_state), reward, messages

    def _trigger_chase(
        self, state: GameState
    ) -> tuple[GameState, float, list[str]]:
        before = state.resources
        after = before.apply(ResourceDelta(tickets=1, team_strength=1))
        reward = (
            self.ruleset.objective.resource_reward(before, after)
            + self.ruleset.objective.chase_penalty
        )
        chased_state = replace(
            state,
            resources=after,
            chase_count=state.chase_count + 1,
        )
        messages = ["行动力耗尽，自动赢得追猎"]
        # A synthetic placeholder node carries only the exit semantics; chase
        # rewards were applied above and do not masquerade as an on-map node.
        placeholder = MapNode(
            node_id="CHASE",
            index=0,
            row=0,
            col=0,
            node_type=NodeType.BATTLE_BOSS,
            distance_from_start=0,
        )
        chased_state, clear_reward, clear_messages = self._advance_floor(
            chased_state, placeholder, chased=True
        )
        reward += clear_reward
        messages.extend(clear_messages)
        return chased_state, reward, messages

    def _must_trigger_chase(self, state: GameState) -> bool:
        if state.terminal or state.pending_node_id is not None:
            return False
        frontier = self.reachable_frontier(state)
        return not frontier or min(frontier.values()) > state.resources.action_points

    def _refresh_revealed(self, state: GameState) -> GameState:
        if state.terminal:
            return state
        revealed = set(state.revealed)
        floor_map = state.floor_map
        if floor_map.floor == 6:
            # Source Confluence is the third-ending special layer.  Unlike
            # normal regions, the game reveals every node immediately.
            revealed.update(node.node_id for node in floor_map.nodes)
            return replace(state, revealed=frozenset(revealed))
        for node in floor_map.nodes:
            if node.node_type in {
                NodeType.START,
                NodeType.FINAL,
                NodeType.EVACUATE,
                NodeType.BATTLE_BOSS,
                NodeType.LIGHT,
                NodeType.DOOR,
            }:
                revealed.add(node.node_id)
        revealed.update(self.reachable_frontier(state))

        adjacency = floor_map.adjacency()
        for light in (
            node
            for node in floor_map.nodes
            if node.node_type is NodeType.LIGHT and node.node_id in state.completed
        ):
            distances = {light.node_id: 0}
            queue = deque([light.node_id])
            while queue:
                node_id = queue.popleft()
                if distances[node_id] >= 2:
                    continue
                for other in adjacency[node_id]:
                    if other not in distances:
                        distances[other] = distances[node_id] + 1
                        queue.append(other)
            revealed.update(distances)
        return replace(state, revealed=frozenset(revealed))
