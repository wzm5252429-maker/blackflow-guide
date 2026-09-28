"""Bounded route search on the observed map, with no reward or RNG previews.

This is a policy approximation. It plans movement and concept activations;
unknown event contents and incoming loot remain unknown. Shopping continues to
use EconomyEvaluator's public quotes and inventory decisions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import heapq

from .agents import EconomyEvaluator, unknown_node_distribution
from .domain import ActionKind, NodeType
from .policy_cache import PublicPolicyCacheMixin
from .policy_constraints import (allowed_action_ids, known_fifth_floor_boss,
    known_unprepared_employ, public_node_has_development, shop_entry_is_prepared)


@dataclass(frozen=True, slots=True)
class RouteSearchConfig:
    depth: int = 6
    beam_width: int = 32
    branches: int = 18
    discount: float = 0.94
    chase_penalty: float = 5.0
    random_continuation_width: int = 8
    expiring_vehicle_waste_penalty: float = 6.0


@dataclass(frozen=True, slots=True)
class RoutePosition:
    node: int
    completed: int
    ap: int
    uses: tuple[int, ...]
    white_birds: int
    white_dogs: int
    revealed: int = 0
    door_ready: bool = False
    visited_shops: int = 0
    ended: bool = False
    terminal_progress: int = 0
    terminal_pending: bool = False
    cash: int = 0
    hp: int = 0
    shield: int = 0
    homing_moves: int = 0


class ObservableRouteEvaluator(PublicPolicyCacheMixin, EconomyEvaluator):
    """Search several actual geometric moves before committing to the first.

    Never call transition to resolve an unentered event, battle or refresh.
    Only the current observation is compiled into this temporary route model.
    Its estimated rewards are not acquisitions or environment state updates.
    """
    def __init__(self, simulator, config=None, route_config=None):
        super().__init__(simulator, config)
        self.route_config = route_config or RouteSearchConfig()
        self._route_key = None
        self._route_action = None
        self._committed_move = None

    def action_score(self, state, action_id):
        exit_menu = bool(state.pending_node_id and state.floor_map.node(
            state.pending_node_id).node_type in (NodeType.FINAL, NodeType.EVACUATE))
        if state.economy_enabled and (not state.pending_node_id or exit_menu) and state.resources.parts <= state.parts_capacity:
            observed = self.observable_state(state)
            key = observed.state_key()
            if key != self._route_key:
                self._route_key = key
                self._route_action = self.plan_route(observed)
            if exit_menu and self._route_action is not None:
                action = self.simulator.decode_action(state, action_id)
                option = self.simulator.available_options(state)[action.option_index or 0]
                if option.operation in ('advance', 'leave'):
                    # Stored recruitment remains preferable to closing this
                    # screen; only compare the two departure choices here.
                    return 5.0 if action_id == self._route_action else -5.0
            elif action_id == self._route_action:
                return 10000.0
        return super().action_score(state, action_id)

    def plan_route(self, state):
        # Defence in depth for callers outside action_score.
        state = self.observable_state(state)
        exit_menu_actions = {}
        if state.pending_node_id and not state.terminal:
            node = state.floor_map.node(state.pending_node_id)
            if node.node_type in (NodeType.FINAL, NodeType.EVACUATE):
                options = self.simulator.available_options(state)
                exit_menu_actions = {options[a.option_index].operation: a.action_id
                    for a in self.simulator.legal_actions(state)
                    if a.option_index is not None and
                    options[a.option_index].operation in ('advance', 'leave')}
                if set(exit_menu_actions) == {'advance', 'leave'}:
                    # Leaving this already-observed menu has no random result.
                    # Its physical node becomes traversable; no reward is paid.
                    state = replace(state, pending_node_id=None,
                        completed=state.completed | {state.current_node_id})
                    self._committed_move = None
                else:
                    exit_menu_actions = {}
        if state.pending_node_id or state.terminal:
            self._committed_move = None
            return None
        # Equipping changes the action menu but reveals no new outcome. Keep
        # the selected first move through that interface change; otherwise the
        # bounded beam can change its pruning and oscillate between vehicles.
        signature = replace(state, equipped_instance_id=None).state_key()
        if self._committed_move is not None:
            prior, item_id, target_id = self._committed_move
            if prior == signature:
                permitted = set(allowed_action_ids(self.simulator, state, self.policy_constraints))
                legal = tuple(a for a in self.simulator.legal_actions(state) if a.action_id in permitted)
                if state.equipped_instance_id == item_id:
                    action = next((a for a in legal if a.kind == ActionKind.MOVE
                        and (target_id is None or a.target_node_id == target_id)), None)
                else:
                    action = next((a for a in legal if a.kind == ActionKind.EQUIP
                        and a.equipment_instance_id == item_id), None)
                if action is not None:
                    return action.action_id
            self._committed_move = None
        config = self.route_config
        if state.portal_context is not None and state.portal_context.variation_id == 5:
            # Blue pond contents reroll after every move. The currently seen
            # shop/wish cannot be held fixed for a later hypothetical move.
            # Commit one observed move, then replan from the actual refresh.
            config = replace(config, depth=min(config.depth, 1))
        task_constraints = getattr(self, 'policy_constraints', None)
        exhaust_final_floor = bool(state.floor == 5 and
            (state.portal_context is None or state.portal_context.outer_action_points <= 0) and
            getattr(task_constraints, 'fifth_floor_exhaust_actions', False))
        nodes = state.floor_map.nodes
        index = {n.node_id: i for i, n in enumerate(nodes)}
        known_final_boss_indices = frozenset(i for i, node in enumerate(nodes)
            if known_fifth_floor_boss(state, node.node_id)) if exhaust_final_floor else frozenset()
        adjacency = state.floor_map.adjacency()
        links = [[(index[j], 1) for j in adjacency[n.node_id]] for n in nodes]
        doors = [i for i, n in enumerate(nodes) if n.node_type == NodeType.DOOR]
        if len(doors) == 2:
            a, b = doors
            links[a].append((b, 0))
            links[b].append((a, 0))
        door_mask = sum(1 << point for point in doors)
        completed = sum(1 << index[n] for n in state.completed if n in index)
        occupied = (state.resident_context.occupied_node_ids if state.resident_context else frozenset())
        occupied_mask = sum(1 << index[n] for n in occupied if n in index)
        strongholds = (state.resident_context.stronghold_node_ids-state.resident_context.defeated_stronghold_ids
            if state.resident_context else frozenset())
        stronghold_mask = sum(1 << index[n] for n in strongholds if n in index)
        overlay_mask = occupied_mask | stronghold_mask
        # An active public resident overrides an earlier completed marker.
        # The first planned defeat, rather than that stale marker, clears it.
        completed &= ~overlay_mask
        equipment = [x for x in state.item_instances if x.category == 'MOVE']
        definitions = [self._definition(x.item_id) for x in equipment]
        initial = RoutePosition(index[state.current_node_id], completed, state.resources.action_points,
            tuple(x.uses_remaining or 0 for x in equipment),
            sum(x.item_id.endswith('P_01') for x in state.item_instances),
            sum(x.item_id.endswith('P_02') for x in state.item_instances),
            sum(1 << index[n] for n in state.revealed if n in index),
            getattr(state, 'door_ready_node_id', None) == state.current_node_id,
            terminal_progress=min(2, max(0, dict(state.event_counters).get(
                'relic_layer:rogue_6_relic_artifact_1', 0))),
            terminal_pending=('rogue_6_relic_artifact_1' in state.inventory
                and 'rogue_6_relic_artifact_2' not in state.inventory
                and dict(state.event_counters).get(
                    'relic_layer:rogue_6_relic_artifact_1:paid:rogue_6_relic_artifact_2', 0) < 1),
            cash=state.resources.gold, hp=state.resources.hp, shield=state.resources.shield)
        painted_wish = sum(x.item_id.endswith('P_05') for x in state.item_instances)
        painted_sacrifice = sum(x.item_id.endswith('P_06') for x in state.item_instances)
        # Hidden nodes expose only their coarse category. Estimate painted
        # arrivals from that public prior, never their simulator-only type.
        # These are policy utilities, not promised or recorded acquisitions.
        painted_arrivals = []
        for node in nodes:
            if not painted_wish and not painted_sacrifice:
                painted_arrivals.append(0.0)
                continue
            distribution = (unknown_node_distribution(state, node, self.simulator.ruleset,
                self.simulator.map_generator.config)
                if node.node_id not in state.revealed and node.node_id not in state.completed
                else {node.node_type: 1.0})
            painted_arrivals.append(painted_wish * distribution.get(NodeType.WISH, 0.0)
                + 2 * painted_sacrifice * distribution.get(NodeType.SACRIFICE, 0.0))
        permitted = set(allowed_action_ids(self.simulator, state, self.policy_constraints))
        legal = tuple(a for a in self.simulator.legal_actions(state) if a.action_id in permitted)
        legal_ids = {a.action_id for a in legal}
        equip_actions = {a.equipment_instance_id: a.action_id for a in legal if a.kind == ActionKind.EQUIP}
        shops = frozenset((NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP))
        preserved = shops | {NodeType.START, NodeType.DOOR, NodeType.LIGHT,
                             NodeType.FINAL, NodeType.EVACUATE, NodeType.BATTLE_BOSS}
        node_values = [self._node_value(state, n) for n in nodes]
        development_nodes = [public_node_has_development(self.simulator, state, n.node_id,
            self.policy_constraints, completed=bool(completed & (1 << i)))
            for i, n in enumerate(nodes)]
        final_chase_value = max((node_values[i] for i, node in enumerate(nodes)
            if node.node_type == NodeType.BATTLE_BOSS), default=12.0) if exhaust_final_floor else 0.0
        walking_cache = {}
        path_cache = {}
        tunnel_arrival_cache = {}
        region = state.region_state
        source_bit = 1 << index[region.source_node_id] if region and region.source_node_id in index else 0
        region_mask = sum(1 << index[n] for n in region.covered_node_ids if n in index) if region else 0
        grass_count = sum(x.item_id.endswith('G_03') for x in state.item_instances)
        blood_mushrooms = sum(x.item_id.endswith('G_02') for x in state.item_instances)
        homing_fruit = sum(x.item_id.endswith('G_10') for x in state.item_instances)
        exit_value_cache = {}

        def exit_rewards(target, remaining_uses, birds, dogs, hp, shield):
            """Price only committing a real exit, with planned charges spent.

            Project already-known consumptions, never any future acquisition
            or destination. The replacement vehicle is valued by the policy's
            declared public prior, not by sampling the actual next reward.
            """
            node = nodes[target]
            if node.node_type not in (NodeType.FINAL, NodeType.EVACUATE):
                return 0.0
            key = node.node_id, tuple(remaining_uses), birds, dogs, hp, shield
            if key not in exit_value_cache:
                uses_by_id = {item.instance_id: count for item, count
                    in zip(equipment, remaining_uses)}
                copies = {'P_01': birds, 'P_02': dogs}
                projected = []
                for item in state.item_instances:
                    if item.category == 'MOVE':
                        count = uses_by_id[item.instance_id]
                        if count > 0:
                            projected.append(replace(item, uses_remaining=count))
                    elif item.item_id[-4:] in copies:
                        suffix = item.item_id[-4:]
                        if copies[suffix] > 0:
                            projected.append(item)
                            copies[suffix] -= 1
                    else:
                        projected.append(item)
                retained_ids = {item.item_id for item in projected}
                removed_ids = {item.item_id for item in state.item_instances} - retained_ids
                removed_parts = len(state.item_instances) - len(projected)
                observed = replace(state, item_instances=tuple(projected),
                    inventory=state.inventory - removed_ids,
                    resources=replace(state.resources,
                        parts=max(0, state.resources.parts-removed_parts), hp=hp, shield=shield),
                    equipped_instance_id=(state.equipped_instance_id if any(
                        item.instance_id == state.equipped_instance_id for item in projected) else None))
                exit_value_cache[key] = (self._exit_vehicle_value(observed, node)
                    + self._exit_health_value(observed, node))
            return exit_value_cache[key]

        def region_active(position):
            return bool(region and region.active and not (region.removable and position.completed & source_bit))

        cleared_shop_values = {}
        cleared_resident_values = {}

        def active_overlays(position):
            # Defeating any stronghold removes all roaming markers. This is
            # a documented deterministic effect, not a prediction of movement.
            roaming = 0 if position.completed & stronghold_mask else occupied_mask
            return (stronghold_mask | roaming) & ~position.completed

        def node_value_at(position, target):
            node = nodes[target]
            if overlay_mask & (1 << target) and not active_overlays(position) & (1 << target):
                key = target, region_active(position), bool(position.completed & (1 << target))
                if key not in cleared_resident_values:
                    cleared_resident_values[key] = self._node_value(projected_state(position),
                        replace(node, node_type=kind_at(position, target)))
                return cleared_resident_values[key]
            if (region and region.active and region.removable and not region_active(position)
                    and region.ideology in ('储藏室', '“储藏室”', 'storage_room')
                    and node.node_type in shops):
                if target not in cleared_shop_values:
                    cleared = replace(state, region_state=replace(region, active=False))
                    # Keep actual observed stock and lost_slots. Clearing the
                    # source improves sale terms and future ungenerated stock;
                    # it never restores slots already removed from a shop.
                    cleared_shop_values[target] = self._node_value(cleared, node)
                return cleared_shop_values[target]
            return node_values[target]

        def kind_at(position, target):
            kind = nodes[target].node_type
            if position.completed & overlay_mask & (1 << target):
                return NodeType.EMPTY
            return NodeType.EMPTY if position.completed & (1 << target) and kind not in preserved else kind

        projected_state_cache = {}

        def projected_state(position):
            key = (position.node, position.completed, position.ap, position.cash,
                position.hp, position.shield, position.homing_moves)
            if key not in projected_state_cache:
                projected = replace(state, current_node_id=nodes[position.node].node_id,
                    completed=frozenset(node.node_id for i, node in enumerate(nodes)
                        if position.completed & (1 << i)),
                    resources=replace(state.resources, action_points=position.ap,
                        gold=position.cash, hp=position.hp, shield=position.shield))
                if position.homing_moves:
                    projected = replace(projected, item_instances=tuple(replace(item,
                        appraisal=max(0, item.appraisal-2*position.homing_moves))
                        if item.item_id.endswith('G_10') else item for item in projected.item_instances))
                if region and not region_active(position):
                    projected = replace(projected, region_state=replace(region, active=False))
                cleared = position.completed & overlay_mask
                if cleared and projected.resident_context:
                    residents = projected.resident_context
                    defeated = frozenset(node.node_id for i, node in enumerate(nodes)
                        if position.completed & stronghold_mask & (1 << i))
                    projected = replace(projected, resident_context=replace(residents,
                        markers=tuple(marker for marker in residents.markers
                            if not defeated and (marker.node_id not in index
                                or not position.completed & (1 << index[marker.node_id]))),
                        defeated_stronghold_ids=residents.defeated_stronghold_ids | defeated))
                    layouts = list(projected.maps)
                    layouts[projected.floor_index] = replace(projected.floor_map, nodes=tuple(
                        replace(node, node_type=NodeType.EMPTY) if cleared & (1 << i) else node
                        for i, node in enumerate(nodes)))
                    projected = replace(projected, maps=tuple(layouts))
                projected_state_cache[key] = projected
            return projected_state_cache[key]

        def transit_preference(position, target, gear):
            kind = kind_at(position, target)
            if kind not in (NodeType.EMPTY, NodeType.DOOR) or not any(position.uses):
                return 0.0
            # The shared preference must see consumed rewards as completed,
            # so an old resident/fruit marker cannot exempt every revisit.
            # Future discoveries do not expose their hidden concrete types.
            return self._transit_arrival_preference(projected_state(position),
                replace(nodes[target], node_type=kind),
                equipment=definitions[gear] if gear >= 0 else None,
                vehicle_available=True, white_dogs=position.white_dogs)

        def walking(position):
            cache_key = position.node, position.completed, position.door_ready
            if cache_key in walking_cache:
                return walking_cache[cache_key]
            distance = {position.node: 0}
            paths = {position.node: ()}
            tunnel_arrival = {position.node: False}
            queue = deque((position.node,))
            frontier = {}
            while queue:
                current = queue.popleft()
                for target, cost in links[current]:
                    tunnel = cost == 0
                    if tunnel and current == position.node and not position.door_ready:
                        cost = 1  # Re-enter this endpoint before another transfer.
                    total = distance[current] + cost
                    if target != position.node:
                        frontier[target] = min(total, frontier.get(target, total))
                    if total < distance.get(target, 10**9):
                        distance[target] = total
                        paths[target] = paths[current] + (target,)
                        tunnel_arrival[target] = tunnel
                        if (position.completed | door_mask) & (1 << target) and not active_overlays(position) & (1 << target):
                            queue.appendleft(target) if cost == 0 else queue.append(target)
            if kind_at(position, position.node) in shops | {NodeType.FINAL, NodeType.EVACUATE}:
                frontier[position.node] = 1
            walking_cache[cache_key] = frontier
            paths[position.node] = (position.node,)
            path_cache[cache_key] = paths
            tunnel_arrival_cache[cache_key] = tunnel_arrival
            return frontier

        def raw_moves(position):
            for target, cost in walking(position).items():
                if cost <= position.ap:
                    yield target, cost, -1
            origin = nodes[position.node]
            for gear, definition in enumerate(definitions):
                if position.uses[gear] <= 0 or definition.random_move:
                    continue  # Uncontrollable destinations retain the baseline fallback.
                cost = int(definition.move_ap or 0)
                if position.uses[gear] == 1 and 'rogue_6_relic_cargo_6' in state.inventory:
                    cost = 0
                if cost > position.ap:
                    continue
                for target, node in enumerate(nodes):
                    kind = kind_at(position, target)
                    if target == position.node and kind not in shops:
                        continue
                    if definition.move_range and (node.row-origin.row, node.col-origin.col) not in definition.move_range:
                        continue
                    allowed = definition.move_target_types
                    battle = kind in (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE, NodeType.BATTLE_BOSS, NodeType.BATTLE_SAVAGE)
                    if 'ALL' not in allowed and not ('EVENTS' in allowed and not battle) and kind.value not in allowed:
                        continue
                    if definition.item_id.endswith('M_10') and not position.revealed & (1 << target):
                        continue
                    yield target, cost, gear

        def random_landings(position):
            candidates = [target for target in range(len(nodes))
                if kind_at(position, target) not in (
                    NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE,
                    NodeType.BATTLE_BOSS, NodeType.BATTLE_SAVAGE)]
            if self.simulator.economy.config.random_transport_unknown_first:
                unknown = [target for target in candidates if not position.revealed & (1 << target)]
                if unknown:
                    candidates = unknown
            return candidates

        def develops(position, target, gear):
            kind = kind_at(position, target)
            if gear >= 0 and position.white_dogs and kind not in (
                    NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE,
                    NodeType.BATTLE_BOSS, NodeType.BATTLE_SAVAGE):
                return True
            development = development_nodes[target]
            if overlay_mask & (1 << target):
                development = public_node_has_development(self.simulator, projected_state(position),
                    nodes[target].node_id, self.policy_constraints,
                    completed=bool(position.completed & (1 << target)))
            if not development or (position.completed & (1 << target) and kind not in shops):
                return False
            if kind == NodeType.PORTAL:
                return (sum(position.uses) - int(gear >= 0) > 0
                    or gear >= 0 and position.uses[gear] == 1 and 'rogue_6_relic_cargo_5' in state.inventory)
            return True

        arrival_state_cache = {}

        def shop_arrival_state(position, target, cost, gear):
            key = position, target, cost, gear
            if key in arrival_state_cache:
                return arrival_state_cache[key]
            projected = projected_state(position)
            cash = position.cash
            if region_active(position) and region.policy in ('改良', 'improve'):
                path = (target,) if gear >= 0 else path_cache[position.node, position.completed, position.door_ready][target]
                cash = max(0, cash-2*sum(bool(region_mask & (1 << point)) for point in path))
            if gear >= 0 and definitions[gear].item_id.endswith('M_10'):
                cash += 4
            if homing_fruit and (gear >= 0 or cost > 0):
                projected = replace(projected, item_instances=tuple(replace(item,
                    appraisal=max(0, item.appraisal-2)) if item.item_id.endswith('G_10')
                    else item for item in projected.item_instances))
            projected = replace(projected, current_node_id=nodes[target].node_id,
                resources=replace(projected.resources, gold=cash))
            arrival_state_cache[key] = projected
            return projected

        def prepared_shop_arrival(position, target, cost, gear):
            node = replace(nodes[target], node_type=kind_at(position, target))
            if node.node_id not in state.revealed or node.node_type not in shops:
                return True  # Never look through an unknown marker.
            projected = shop_arrival_state(position, target, cost, gear)
            # Only actual money and currently held saleable goods count.
            # Positive appraisal, generated shop stock and arrival rewards
            # are not promised capital for a later hypothetical visit.
            return shop_entry_is_prepared(self.simulator, projected, node,
                arrival_gold=projected.resources.gold, constraints=self.policy_constraints)

        def moves(position):
            # A declared employment preparation rule applies to controllable
            # arrivals only. M07 must retain every real random landing.
            options = [option for option in raw_moves(position)
                if not known_unprepared_employ(projected_state(position), nodes[option[0]].node_id,
                    self.policy_constraints, completed=bool(position.completed & (1 << option[0])))
                and prepared_shop_arrival(position, *option)]
            if (exhaust_final_floor and state.portal_context is None and position.ap > 0
                    and getattr(task_constraints, 'first_ending_only', False)):
                # A normal boss entered on the last AP still ends normally;
                # AP==0 does not turn that battle into a chase. Apply the same
                # preference inside the search, not only to the root action.
                alternatives = [option for option in options if option[0] not in known_final_boss_indices]
                development_alternative = any(develops(position, target, gear)
                    for target, _, gear in alternatives)
                random_alternative = (any(definition.random_move and position.uses[gear] > 0
                    and int(definition.move_ap or 0) <= position.ap
                    and any(develops(position, target, gear) for target in random_landings(position))
                    for gear, definition in enumerate(definitions)))
                if development_alternative or random_alternative:
                    options = alternatives
            yield from options

        def advance(position, target, cost, gear, *, stay_at_exit=False):
            node = nodes[target]
            kind = kind_at(position, target)
            fresh = not position.completed & (1 << target)
            resident_battle = bool(active_overlays(position) & (1 << target))
            battle = kind in (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE, NodeType.BATTLE_BOSS, NodeType.BATTLE_SAVAGE)
            value = node_value_at(position, target)
            gain = value if fresh else 0.0
            if (exhaust_final_floor and kind == NodeType.EMPTY and not development_nodes[target]):
                # Plain ground has no node reward. Real concepts, resident or
                # marked rewards and useful onward movement are priced below.
                gain = 0.0
            gain += transit_preference(position, target, gear)
            terminal_progress, terminal_pending = position.terminal_progress, position.terminal_pending
            if kind in shops:
                if not resident_battle:
                    gain = value if not position.visited_shops & (1 << target) else 2 * self._gold_value(state)
                if terminal_pending:
                    # Actual arrivals, including a paid re-entry to the same
                    # shop, advance the public counter. This is one unique
                    # collectible for the whole route, not one per shop.
                    terminal_progress = min(2, terminal_progress + 1)
                    if terminal_progress >= 2:
                        gain += self.config.relic_value
                        terminal_pending = False
            # Shared node valuation prices an ordinary sacrifice exchange at
            # two utility points, then separately adds useful part conversion,
            # marked fruit rewards and the pond's actual completion bounty.
            uses = list(position.uses)
            birds, dogs = position.white_birds, position.white_dogs
            cash = position.cash
            hp, shield = position.hp, position.shield
            movement_income = 0
            ap = position.ap - cost + int(kind == NodeType.LIGHT and fresh)
            if gear >= 0:
                uses[gear] -= 1
                definition = definitions[gear]
                if definition.item_id.endswith('M_12'):
                    ap += 3
                elif definition.item_id.endswith('M_10'):
                    gain += 4 * self._gold_value(state)
                    movement_income = 4
                if battle:
                    gain += birds * self.config.relic_value
                    birds = 0
                else:
                    gain += dogs * self.config.relic_value
                    dogs = 0
                if fresh:
                    gain += self.config.relic_value * painted_arrivals[target]
                # Value the used charge, without pretending to know replacement loot.
                gain -= 0.5
            replacement_on_break = (gear >= 0 and uses[gear] == 0 and
                'rogue_6_relic_cargo_5' in state.inventory)
            if kind == NodeType.PORTAL and not resident_battle and not any(uses) and not replacement_on_break:
                # A vehicle that breaks on arrival cannot also pay the Black
                # Pond's separate entry sacrifice. Recheck the planned
                # inventory after this move, not only the starting inventory.
                gain -= value if fresh else 0.0
            gain -= cost * self.config.action_point_value
            if fresh and (battle or resident_battle):
                gain += blood_mushrooms * 2 * self._gold_value(state)
            if gear >= 0 or cost > 0:
                gain -= homing_fruit * 2 * self._gold_value(state)
            if not fresh and kind not in shops and ap == position.ap:
                gain -= 0.05  # Break cost-free geometry ties, not an environment cost.
            actual_exit = kind in (NodeType.FINAL, NodeType.EVACUATE, NodeType.BATTLE_BOSS) and not resident_battle
            ended = (actual_exit and not stay_at_exit) or ap <= 0
            terminal_ground = (exhaust_final_floor and state.portal_context is None
                and ended and kind == NodeType.EMPTY)
            if stay_at_exit and fresh:
                # The exit's reward is only obtained by choosing to advance.
                gain -= value
            if actual_exit and not stay_at_exit:
                gain += 0.0 if exhaust_final_floor else 5.0
                if kind == NodeType.EVACUATE:
                    gain += ap * self.config.action_point_value
                elif ap and not exhaust_final_floor:
                    gain -= ap * 0.5
            elif ap <= 0:
                # The pond has its own AP. Exhausting it restores the outer
                # map's saved AP, and causes a chase only if that AP is zero.
                causes_chase = (state.portal_context is None or
                    state.portal_context.outer_action_points <= 0)
                if causes_chase:
                    gain -= 0.0 if exhaust_final_floor else config.chase_penalty
                # On V, consuming AP at a development node still gives the
                # ensuing first-ending boss chase. Its value is not lost by
                # taking the extra node; do not falsely prefer direct boss entry.
                gain += final_chase_value
            if ended and state.portal_context is None and kind != NodeType.PORTAL and not exhaust_final_floor:
                # These charges disappear on an ordinary region change.
                # Prefer a useful free move now to carrying an unusable charge
                # through the exit; this is a policy cost, never extra AP/loot.
                gain -= config.expiring_vehicle_waste_penalty * sum(
                    uses[i] for i, definition in enumerate(definitions)
                    if definition.expires_on_floor_change)
            if region_active(position):
                path = (target,) if gear >= 0 else path_cache[position.node, position.completed, position.door_ready][target]
                affected = sum(bool(region_mask & (1 << point)) for point in path)
                if region.policy in ('改良', 'improve'):
                    paid = min(cash, affected * 2)
                    cash -= paid
                    gain -= paid * self._gold_value(state)
                elif region.policy in ('修正', 'correct'):
                    lost = min(shield, affected)
                    shield -= lost
                    gain -= lost * 0.1
                elif region.policy in ('激进', 'radical'):
                    lost = min(max(0, hp - 1), affected)
                    gain += self._travel_health_value(projected_state(position), node,
                        lost, action_points_after=ap)
                    hp -= lost
                elif region.policy in ('增益', 'benefit') and not terminal_ground:
                    gain += affected * 2 * sum(x.category == 'GOODS' for x in state.item_instances) * self._gold_value(state)
                if fresh:
                    gain += self._region_source_priority(state, node, remaining_action_points=ap)
            if actual_exit and not stay_at_exit:
                # Regional path damage happens before the real exit reward.
                # Its nonlinear health value depends on the arrival HP, not
                # the HP when the route search first started.
                gain += exit_rewards(target, uses, birds, dogs, hp, shield)
            # The runtime charges regional travel costs before M10 pays out.
            cash += movement_income
            # Capital actually on hand, after public travel income/fees.
            # Future sales, bank withdrawals and unknown loot are not cash.
            if not resident_battle:
                arrival_state = (shop_arrival_state(position, target, cost, gear)
                    if kind in shops else projected_state(position))
                gain -= self._green_entry_preparation_penalty(arrival_state,
                    replace(node, node_type=kind), cash)
            after = RoutePosition(target, position.completed | (1 << target), ap,
                tuple(uses), birds, dogs, position.revealed,
                kind == NodeType.DOOR and (gear >= 0 or not tunnel_arrival_cache[position.node, position.completed, position.door_ready].get(target, False)),
                position.visited_shops | ((1 << target) if kind in shops and not resident_battle else 0), ended,
                terminal_progress=terminal_progress, terminal_pending=terminal_pending,
                cash=cash, hp=hp, shield=shield,
                homing_moves=position.homing_moves+int(bool(homing_fruit) and (gear >= 0 or cost > 0)))
            # New contents stay unknown in this route model. Reward only the
            # opportunity to observe them and actual held grass appraisal.
            natural_reveal = sum(1 << point for point in walking(after))
            if kind == NodeType.LIGHT and fresh:
                radius = 3 if self.simulator.economy.config.full_tech else 2
                natural_reveal |= sum(1 << i for i, other in enumerate(nodes)
                    if abs(other.row-node.row) + abs(other.col-node.col) <= radius)
            if region_active(after) and region.ideology in ('弥散虚雾', '“弥散虚雾”'):
                natural_reveal &= ~region_mask
            observed = position.revealed | natural_reveal | (1 << target)
            discoveries = (observed & ~position.revealed).bit_count()
            information_value = (0.0 if exhaust_final_floor and after.ended else self.config.reveal_node_value)
            appraisal_value = 0.0 if terminal_ground else grass_count * self._gold_value(state)
            gain += discoveries * (information_value + appraisal_value)
            if (exhaust_final_floor and after.ended and kind == NodeType.EMPTY
                    and not develops(position, target, gear) and (not grass_count or terminal_ground)):
                gain -= 0.05  # Prefer finishing over an otherwise valueless final detour.
            return replace(after, revealed=observed), gain

        def future_potential(position):
            origin = nodes[position.node]
            best = 0.0
            for target, node in enumerate(nodes):
                if position.completed & (1 << target) or node.node_id not in state.revealed:
                    continue
                extra = (painted_wish if node.node_type == NodeType.WISH else
                         2 * painted_sacrifice if node.node_type == NodeType.SACRIFICE else 0)
                if extra:
                    distance = abs(origin.row-node.row) + abs(origin.col-node.col)
                    best = max(best, extra * self.config.relic_value / (1 + distance))
            return best

        def search(start, depth_limit, width, *, root_actions):
            beam = [(0.0, start, None if root_actions else -1, None)]
            best_by_first = {}
            moves_by_first = {}
            for depth in range(depth_limit):
                expanded = []
                dedup = {}
                for score, position, first, first_move in beam:
                    if position.ended or position.ap < 0:
                        continue
                    options = []
                    for target, cost, gear in moves(position):
                        first_action = first
                        selected_move = first_move
                        if first is None:
                            item_id = equipment[gear].instance_id if gear >= 0 else None
                            first_action = nodes[target].index if item_id == state.equipped_instance_id else equip_actions.get(item_id)
                            if first_action not in legal_ids:
                                continue
                            selected_move = target, gear
                        continuations = (False, True) if (kind_at(position, target) in (
                            NodeType.FINAL, NodeType.EVACUATE) and not active_overlays(position) & (1 << target)
                            and position.ap > cost) else (False,)
                        for stay in continuations:
                            after, gain = advance(position, target, cost, gear, stay_at_exit=stay)
                            new_score = score + config.discount**depth * gain
                            options.append((new_score + 0.25 * future_potential(after), new_score, after, first_action, selected_move))
                    for _, new_score, after, first_action, selected_move in heapq.nlargest(config.branches, options, key=lambda row: row[0]):
                        # Keep each possible first decision independently.
                        key = after, first_action
                        if new_score <= dedup.get(key, -float('inf')):
                            continue
                        dedup[key] = new_score
                        if new_score > best_by_first.get(first_action, -float('inf')):
                            best_by_first[first_action] = new_score
                            moves_by_first[first_action] = selected_move
                        expanded.append((new_score, after, first_action, selected_move))
                if not expanded:
                    break
                beam = heapq.nlargest(width, expanded,
                    key=lambda row: row[0] + 0.25 * future_potential(row[1]))
            return best_by_first, moves_by_first

        best_by_first, moves_by_first = search(initial, config.depth, config.beam_width, root_actions=True)
        # A random vehicle is a chance decision, never a selectable destination.
        # Average *every* observable possible landing, including bad exits; do
        # not maximize over outcomes or resolve the environment's private RNG.
        # The public sinister/mysterious marker exposes battle/nonbattle
        # eligibility even before exact node information has been revealed.
        chance_cache = {}
        eligible_landings = [(target, nodes[target]) for target in random_landings(initial)]
        for gear, definition in enumerate(definitions):
            if not definition.random_move or initial.uses[gear] <= 0:
                continue
            item_id = equipment[gear].instance_id
            first_action = (next((a.action_id for a in legal if a.kind == ActionKind.MOVE), None)
                if item_id == state.equipped_instance_id else equip_actions.get(item_id))
            cost = int(definition.move_ap or 0)
            if first_action not in legal_ids or cost > initial.ap:
                continue
            total = value = 0.0
            for target, node in eligible_landings:
                continuations = (False, True) if (kind_at(initial, target) in (
                    NodeType.FINAL, NodeType.EVACUATE) and not active_overlays(initial) & (1 << target)
                    and initial.ap > cost) else (False,)
                landing_values = []
                for stay in continuations:
                    after, gain = advance(initial, target, cost, gear, stay_at_exit=stay)
                    # Narrower continuation limits computation without selecting
                    # only fortunate random landings. The player can choose to
                    # stay after seeing an exit, so maximize only that real choice.
                    if after not in chance_cache:
                        continuation, _ = search(after, max(0, config.depth - 1),
                            config.random_continuation_width, root_actions=False)
                        chance_cache[after] = max(continuation.values(), default=0.0)
                    landing_values.append(gain + config.discount * chance_cache[after])
                value += max(landing_values)
                total += 1.0
            if total:
                best_by_first[first_action] = value / total
                moves_by_first[first_action] = None, gear
        if exit_menu_actions:
            kind = nodes[initial.node].node_type
            direct_value = (0.0 if exhaust_final_floor else 5.0)
            direct_value += exit_rewards(initial.node, initial.uses,
                initial.white_birds, initial.white_dogs, initial.hp, initial.shield)
            direct_value += (initial.ap * self.config.action_point_value if kind == NodeType.EVACUATE
                else 0.0 if exhaust_final_floor else -initial.ap * 0.5)
            if not exhaust_final_floor:
                direct_value -= config.expiring_vehicle_waste_penalty * sum(
                    initial.uses[i] for i, definition in enumerate(definitions)
                    if definition.expires_on_floor_change)
            continue_value = max(best_by_first.values(), default=-float('inf'))
            return exit_menu_actions['leave' if initial.ap > 0 and continue_value > direct_value else 'advance']
        if not best_by_first:
            return None
        chosen = max(best_by_first, key=lambda a: (best_by_first[a], -a))
        if any(a.action_id == chosen and a.kind == ActionKind.EQUIP for a in legal):
            target, gear = moves_by_first[chosen]
            self._committed_move = (signature,
                equipment[gear].instance_id if gear >= 0 else None,
                nodes[target].node_id if target is not None else None)
        return chosen
