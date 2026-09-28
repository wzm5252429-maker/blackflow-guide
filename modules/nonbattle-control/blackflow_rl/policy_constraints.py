"""First-ending task scope and optional historical strategy restrictions.

The simulator keeps the real options and prices. Controllers share this mask;
no item is deleted, re-rolled, or replaced to manufacture an allowed outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import inspect
from pathlib import Path

from .domain import ActionKind, EventOption, ItemInstance, NodeType, ResourceDelta


FORBIDDEN_RELIC_IDS = frozenset(("rogue_6_relic_final_1", "rogue_6_relic_final_2", "rogue_6_relic_final_3"))
FORBIDDEN_ITEM_FLAGS = FORBIDDEN_RELIC_IDS | {"alpha", "beta", "beacon", "ending_3_key"}
LAKE_OFFERING_CHOICES = frozenset(("choice_ro6_bat5_3", "choice_ro6_bat5_7", "choice_ro6_bat5_8"))


@dataclass(frozen=True, slots=True)
class UserPolicyConstraints:
    first_ending_only: bool = True
    minimum_ingot_reserve: int = 3
    fifth_floor_exhaust_actions: bool = True
    first_floor_full_withdrawal: bool = True
    second_floor_green_entry_gold: int = 12
    preserve_redmoss_tool_through_floor: int = 3
    employ_minimum_corn_copies: int = 2
    minimum_shop_entry_capital: int = 12

    @property
    def strategy_advice_enabled(self):
        """Whether any optional manual strategy restriction is configured."""
        return bool(self.minimum_ingot_reserve or self.fifth_floor_exhaust_actions
            or self.first_floor_full_withdrawal or self.second_floor_green_entry_gold
            or self.preserve_redmoss_tool_through_floor or self.employ_minimum_corn_copies
            or self.minimum_shop_entry_capital)

    @property
    def sha256(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()


DEFAULT_POLICY_CONSTRAINTS = UserPolicyConstraints()
# Retain DEFAULT for reproducing the historical manual-policy experiments.
# Neural policies opt into only game legality and the first-ending task scope.
AUTONOMOUS_POLICY_CONSTRAINTS = UserPolicyConstraints(
    minimum_ingot_reserve=0,
    fifth_floor_exhaust_actions=False,
    first_floor_full_withdrawal=False,
    second_floor_green_entry_gold=0,
    preserve_redmoss_tool_through_floor=0,
    employ_minimum_corn_copies=0,
    minimum_shop_entry_capital=0,
)


def public_resident_replaces_node(state, node_id):
    """Public battle overlays take precedence over the underlying node menu."""
    residents = state.resident_context
    return bool(residents and (node_id in getattr(residents, 'occupied_node_ids', frozenset())
        or node_id in getattr(residents, 'stronghold_node_ids', frozenset())
            -getattr(residents, 'defeated_stronghold_ids', frozenset())))


def employ_entry_is_prepared(state, constraints=DEFAULT_POLICY_CONSTRAINTS):
    """Count actual held corn, never inventory flags or abstract part totals."""
    return sum(item.item_id == 'rogue_6_scrap_G_04' and item.category == 'GOODS'
        for item in state.item_instances) >= getattr(constraints, 'employ_minimum_corn_copies', 2)


def known_unprepared_employ(state, node_id, constraints=DEFAULT_POLICY_CONSTRAINTS, *, completed=None):
    """An intentional fresh entry, using only a revealed node's public type.

    A completed node is transit: EconomyEngine.enter returns before opening
    another employment menu. Unknown destinations are never classified here.
    The caller must not use this helper to censor random transport outcomes.
    """
    done = node_id in state.completed if completed is None else completed
    return (getattr(state, 'economy_enabled', False) and not done
        and node_id in state.revealed
        and state.floor_map.node(node_id).node_type == NodeType.EMPLOY
        and not public_resident_replaces_node(state, node_id)
        and not employ_entry_is_prepared(state, constraints))


def _intentional_employ_move_is_allowed(simulator, state, action, constraints):
    if action.kind != ActionKind.MOVE:
        return True  # Equipment, pending menus and forced arrivals stay real.
    if action.equipment_instance_id:
        item = next(item for item in state.item_instances
            if item.instance_id == action.equipment_instance_id)
        if simulator.economy.catalog.items[item.item_id].random_move:
            return True  # Its action target is an anchor, not a chosen result.
    return not known_unprepared_employ(state, action.target_node_id, constraints)


REDMOSS_EVENT_NAME = "呼吸的红苔"
REDMOSS_TOOL_IDS = frozenset(("rogue_6_scrap_G_01", "rogue_6_scrap_G_12"))


def redmoss_preparation_active(state, constraints=DEFAULT_POLICY_CONSTRAINTS):
    return (getattr(state, "economy_enabled", False) and not state.terminal
        and state.floor <= getattr(constraints, "preserve_redmoss_tool_through_floor", 3)
        and REDMOSS_EVENT_NAME not in state.seen_event_names)


def last_redmoss_tool(state, instance, constraints=DEFAULT_POLICY_CONSTRAINTS):
    return bool(instance and instance.item_id in REDMOSS_TOOL_IDS and redmoss_preparation_active(state, constraints)
        and sum(item.item_id in REDMOSS_TOOL_IDS for item in state.item_instances) == 1)


def _shop_saleable_goods(simulator, state, node, constraints):
    if node.node_type != NodeType.SCRAP_SHOP and not simulator.economy.config.full_tech:
        return ()
    goods = tuple(item for item in state.item_instances if item.category == 'GOODS')
    # Multiple preparation tools may be sold, but not all of them together.
    tools = tuple(item for item in goods if item.item_id in REDMOSS_TOOL_IDS)
    protected = (max(tools, key=lambda item: (simulator.economy.quote_sell(state, item), item.instance_id))
        if tools and redmoss_preparation_active(state, constraints) else None)
    return tuple(item for item in goods if item is not protected)


def _project_shop_goods_sale(state, item, quote):
    remaining = tuple(other for other in state.item_instances if other.instance_id != item.instance_id)
    return replace(state, item_instances=remaining,
        inventory=(state.inventory if any(other.item_id == item.item_id for other in remaining)
            else state.inventory-{item.item_id}),
        resources=replace(state.resources, gold=state.resources.gold+quote,
            parts=max(0, state.resources.parts-1)))


def _known_vine_first_trade(simulator, state, node, shop, goods, constraints):
    """Conservative executable first purchase, not a loan against vine growth.

    Only actual pre-existing vines gain appraisal. Sales fund the first buy;
    later appraisal growth contributes profit only through its real sell quote.
    Entry gifts, banking, three-sale bonuses and unknown stock are excluded.
    """
    engine = simulator.economy
    vines = tuple(item for item in state.item_instances if item.item_id == 'rogue_6_scrap_G_09'
        and item.category == 'GOODS')
    if not vines or not goods:
        # Cash alone can fund a purchase even when there are no saleable goods;
        # a held vine is itself saleable at an actual shop, so no case is lost.
        return False
    for slot in shop.stock:
        if slot.sold or slot.item_id.startswith('service:') or slot.item_id in state.inventory:
            continue
        definition = engine.catalog.items[slot.item_id]
        if definition.category not in {'MOVE', 'GOODS', 'PASSIVE'}:
            continue
        price = engine.quote_buy(state, definition, base_price=slot.price)
        option = EventOption('buy:'+slot.slot_id, definition.name, ResourceDelta(gold=-price),
            operation='purchase', item_id=slot.item_id, price=price, ends_node=False)
        funded = state
        # Preserve vines whenever other actual natural objects can fund the
        # transaction. Selling a vine removes its subsequent growth benefit.
        ordered = sorted(goods, key=lambda item: (item.item_id == 'rogue_6_scrap_G_09',
            -engine.quote_sell(state, item), item.instance_id))
        required = price+max(3, preparation_cash_target(state, constraints))
        for item in ordered:
            if funded.resources.gold >= required and funded.resources.parts <= funded.parts_capacity:
                break
            funded = _project_shop_goods_sale(funded, item, engine.quote_sell(funded, item))
        if (funded.resources.gold < required or funded.resources.parts > funded.parts_capacity
                or not option.is_available(funded.resources, funded.inventory)
                or not option_is_allowed(funded, option, constraints)):
            continue
        retained = tuple(item for item in funded.item_instances
            if item.category == 'GOODS' and item.item_id == 'rogue_6_scrap_G_09')
        if not retained:
            continue
        acquisitions = 4 if slot.item_id == 'rogue_6_scrap_G_07' else 1
        growth_quote = sum(engine.quote_sell(funded, replace(item, appraisal=item.appraisal+4*acquisitions))
            -engine.quote_sell(funded, item) for item in retained)
        purchased = ItemInstance('policy_first_trade', slot.item_id, definition.category,
            definition.move_uses or definition.passive_uses, int(definition.sell_price or 0))
        if acquisitions == 4:
            purchased = replace(purchased, appraisal=purchased.appraisal+3)
        resale = engine.quote_sell(funded, purchased)
        if acquisitions == 4:
            moss = engine.catalog.items['rogue_6_scrap_G_08']
            resale += 3*engine.quote_sell(funded, ItemInstance('policy_ball', moss.item_id,
                'GOODS', appraisal=int(moss.sell_price or 0)))
        if growth_quote > price-resale:
            return True
    return False


def shop_entry_details(simulator, state, node, *, arrival_gold=None, constraints=DEFAULT_POLICY_CONSTRAINTS):
    """Public entry capital; the threshold is policy, not a game item price.

    Callers may supply publicly projected arrival cash and held instances.
    Known negative changes such as G10 depreciation must already be applied;
    future appraisal gains and shop-entry gifts need not be assumed.
    """
    cash = max(0, state.resources.gold if arrival_gold is None else arrival_gold)
    threshold = max(0, getattr(constraints, 'minimum_shop_entry_capital', 12))
    reason = ('not_economy' if not state.economy_enabled else
        'unknown_node' if node.node_id not in state.revealed else
        'not_shop' if node.node_type not in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP} else
        'resident_override' if public_resident_replaces_node(state, node.node_id) else None)
    details = dict(cash=cash, goods_quote=0, threshold=threshold, known_stock=False,
        vine_first_trade_ready=False, applies=reason is None, prepared=True, reason=reason)
    if reason is not None:
        return details  # Do not read a hidden subtype or generated shop stock.
    projected = replace(state, current_node_id=node.node_id, pending_node_id=node.node_id,
        resources=replace(state.resources, gold=cash))
    goods = _shop_saleable_goods(simulator, projected, node, constraints)
    details['goods_quote'] = sum(simulator.economy.quote_sell(projected, item) for item in goods)
    shop = next((shop for shop in state.shops if shop.node_id == node.node_id and shop.visits > 0), None)
    details['known_stock'] = shop is not None
    details['prepared'] = cash+details['goods_quote'] >= threshold
    if not details['prepared'] and shop is not None:
        details['vine_first_trade_ready'] = _known_vine_first_trade(simulator, projected, node, shop, goods, constraints)
        details['prepared'] = details['vine_first_trade_ready']
    details['reason'] = ('vine_first_trade' if details['vine_first_trade_ready'] else
        'cash_and_goods' if details['prepared'] else 'insufficient_entry_capital')
    return details


def shop_entry_is_prepared(simulator, state, node, *, arrival_gold=None, constraints=DEFAULT_POLICY_CONSTRAINTS):
    return shop_entry_details(simulator, state, node, arrival_gold=arrival_gold, constraints=constraints)['prepared']


def shop_move_entry_details(simulator, state, action, constraints=DEFAULT_POLICY_CONSTRAINTS):
    """Audit the same public pre-entry projection used by the actual MOVE mask.

    This does not execute a move or credit its gift/bank income. Random move
    outcomes and unknown node types are deliberately not consulted.
    """
    exempt = dict(cash=state.resources.gold, goods_quote=0,
        threshold=max(0, getattr(constraints, 'minimum_shop_entry_capital', 12)),
        known_stock=False, vine_first_trade_ready=False, applies=False, prepared=True)
    if action.kind != ActionKind.MOVE:
        return dict(exempt, reason='not_move')
    if not state.economy_enabled:
        return dict(exempt, reason='not_economy')
    gear = next((item for item in state.item_instances if item.instance_id == action.equipment_instance_id), None)
    if gear and simulator.economy.catalog.items[gear.item_id].random_move:
        return dict(exempt, reason='random_destination')
    if action.target_node_id not in state.revealed:
        return dict(exempt, reason='unknown_node')
    node = state.floor_map.node(action.target_node_id)
    if node.node_type not in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP}:
        return dict(exempt, reason='not_shop')
    region = state.region_state
    trajectory = (node.node_id,) if gear else action.traversed_node_ids
    fee = (2*sum(region.affects(node_id) for node_id in trajectory)
        if region and region.policy in {'改良', 'improve'} else 0)
    cash = max(0, state.resources.gold-fee)+(4 if gear and gear.item_id == 'rogue_6_scrap_M_10' else 0)
    projected = replace(state, item_instances=tuple(replace(item, appraisal=max(0, item.appraisal-2))
        if item.item_id == 'rogue_6_scrap_G_10' else item for item in state.item_instances))
    return shop_entry_details(simulator, projected, node, arrival_gold=cash, constraints=constraints)


def _intentional_shop_move_is_allowed(simulator, state, action, constraints):
    return shop_move_entry_details(simulator, state, action, constraints)['prepared']


def preparation_cash_target(state, constraints=DEFAULT_POLICY_CONSTRAINTS):
    """Cash kept by discretionary spending before the floor-II parts shop.

    A floor-I tax is floor(gold/10); 13 therefore arrives as 12. No unknown
    future path cost is invented. Actual observed path costs remain in routing.
    """
    base = constraints.minimum_ingot_reserve
    target = getattr(constraints, "second_floor_green_entry_gold", 12)
    if not getattr(state, "economy_enabled", False) or state.floor > 2 or target <= base:
        return base
    if state.floor == 1:
        amount = target
        while amount-amount//10 < target:
            amount += 1
        return max(base, amount)
    if state.pending_node_id and state.floor_map.node(state.pending_node_id).node_type.value == "SCRAP_SHOP":
        return base
    return max(base, target)


def compulsory_bank_options(state, options, constraints=DEFAULT_POLICY_CONSTRAINTS):
    if (not getattr(state, "economy_enabled", False) or state.floor != 1
            or not getattr(constraints, "first_floor_full_withdrawal", True)):
        return ()
    # The engine only exposes an affordable withdrawal below the real entry
    # quota. Never fabricate funds or loop routes to create another entry.
    return tuple(option for option in options if option.operation == "bank_withdraw"
        and option.is_available(state.resources, state.inventory))


class ConstraintNoLegalAction(RuntimeError):
    """A real menu has no action allowed by the user's current constraints."""


def scope_violations(state, constraints=DEFAULT_POLICY_CONSTRAINTS):
    if not constraints.first_ending_only:
        return []
    violations = []
    forbidden = sorted({entry.item_id for entry in state.ledger if entry.operation == "acquire" and entry.item_id in FORBIDDEN_RELIC_IDS})
    if forbidden:
        violations.append("forbidden ending items acquired: " + ",".join(forbidden))
    if state.floor == 6 or any(entry.floor == 6 for entry in state.ledger):
        violations.append("entered floor VI")
    if state.ending_id != "ro6_ending_1":
        violations.append("did not complete ending one")
    return violations


def _ending_option_is_allowed(option, constraints):
    return not constraints.first_ending_only or not (
        option.operation == "expedition_source" or option.item_id in FORBIDDEN_ITEM_FLAGS
        or FORBIDDEN_ITEM_FLAGS.intersection(option.add_items))


def option_is_allowed(state, option, constraints=DEFAULT_POLICY_CONSTRAINTS):
    if not _ending_option_is_allowed(option, constraints):
        return False
    # Keep the three ingots for the actual three one-ingot lake offerings.
    # Forced movement/weather changes are game outcomes, not selectable buys.
    cost = max(0, -option.effect.gold)
    if option.operation in {"purchase", "refresh"}:
        cost = max(cost, option.price or 0)
    if cost and option.option_id not in LAKE_OFFERING_CHOICES and state.resources.gold-cost < preparation_cash_target(state, constraints):
        return False
    return True


def fifth_floor_chase_preference_active(state, constraints=DEFAULT_POLICY_CONSTRAINTS):
    return (constraints.first_ending_only and constraints.fifth_floor_exhaust_actions
        and state.economy_enabled and state.floor == 5 and state.portal_context is None
        and state.pending_node_id is None and state.resources.action_points > 0)


def known_fifth_floor_boss(state, node_id):
    # The private type of an unrevealed battle node is never consulted.
    return (node_id in state.revealed
        and state.floor_map.node(node_id).node_type == NodeType.BATTLE_BOSS)


def public_node_has_development(simulator, state, node_id, constraints=DEFAULT_POLICY_CONSTRAINTS, *, completed=None):
    """An observable development opportunity, never a promised hidden reward.

    Unknown markers remain exploration opportunities regardless of their real
    subtype. Plain ground and transit alone do not justify delaying the boss.
    Actual occupied ground and marked rewards are separate public exceptions.
    """
    node = state.floor_map.node(node_id)
    done = node_id in state.completed if completed is None else completed
    if known_unprepared_employ(state, node_id, constraints, completed=done):
        return False
    if not done:
        residents = state.resident_context
        if residents and node_id in residents.occupied_node_ids:
            return True
        if dict(state.event_counters).get('hidden_natural:'+node_id, 0) == 1:
            return True
    if node_id == state.floor_map.start_node_id:
        return False
    if node_id not in state.revealed and node_id not in state.completed:
        return not done  # Only the public unexplored marker is consulted.
    if node.node_type in {NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP}:
        config = simulator.economy.config
        # This squad receives a real G08 on each paid physical shop arrival.
        if config.full_tech and config.squad == 'multilateral_trade':
            return True
        counters = dict(state.event_counters)
        if ('rogue_6_relic_artifact_1' in state.inventory
                and 'rogue_6_relic_artifact_2' not in state.inventory
                and not counters.get('relic_layer:rogue_6_relic_artifact_1:paid:rogue_6_relic_artifact_2', 0)):
            return True
        shop = next((shop for shop in state.shops if shop.node_id == node_id), None)
        if shop is None or not shop.visits:
            return True  # Unobserved stock; do not inspect its generated contents.
        if (node.node_type == NodeType.BATTLE_SHOP and config.bank_investment >= 15
                and state.bank_balance >= min(shop.total_withdrawn+1, 7)):
            return True  # A real new entry resets the withdrawal allowance.
        projected = replace(state, current_node_id=node_id)
        return any(option.operation in {'purchase', 'refresh', 'sell', 'cultivate'}
            and option.is_available(projected.resources, projected.inventory)
            and option_is_allowed(projected, option, constraints)
            for option in simulator.economy.shop_options(projected, node))
    if done:
        return False
    if node.node_type == NodeType.PORTAL:
        return any(item.category == 'MOVE' for item in state.item_instances)
    return node.node_type not in {NodeType.EMPTY, NodeType.START, NodeType.DOOR,
        NodeType.FINAL, NodeType.EVACUATE, NodeType.BATTLE_BOSS,
        NodeType.STORY, NodeType.STORY_HIDDEN}


def public_random_move_candidates(simulator, state):
    """Current M07 candidate group, using only public battle/nonbattle labels."""
    eligible = tuple(node for node in state.floor_map.nodes
        if node.node_id != state.current_node_id and not node.is_battle)
    if simulator.economy.config.random_transport_unknown_first:
        unknown = tuple(node for node in eligible if node.node_id not in state.revealed)
        if unknown:
            return unknown
    return eligible


def public_move_has_development(simulator, state, action, constraints=DEFAULT_POLICY_CONSTRAINTS):
    if action.kind != ActionKind.MOVE:
        return False
    if not _intentional_employ_move_is_allowed(simulator, state, action, constraints):
        return False
    if not _intentional_shop_move_is_allowed(simulator, state, action, constraints):
        return False
    projected = state
    definition = None
    if action.equipment_instance_id:
        item = next(item for item in state.item_instances
            if item.instance_id == action.equipment_instance_id)
        definition = simulator.economy.catalog.items[item.item_id]
        if item.uses_remaining == 1 and 'rogue_6_relic_cargo_5' not in state.inventory:
            # A drive that breaks on arrival cannot also be sacrificed to enter
            # a pond. The wheel's real replacement is an available exception.
            projected = replace(state, item_instances=tuple(other for other in state.item_instances
                if other.instance_id != item.instance_id))
    candidates = (public_random_move_candidates(simulator, state) if definition and definition.random_move
        else (state.floor_map.node(action.target_node_id),))
    dogs = bool(action.equipment_instance_id and any(item.item_id == 'rogue_6_scrap_P_02'
        for item in state.item_instances))
    return any((dogs and not node.is_battle)
        or public_node_has_development(simulator, projected, node.node_id, constraints) for node in candidates)


def _prefer_fifth_floor_chase(simulator, state, actions, constraints):
    """Avoid a known boss only for an observable development opportunity.

    Equipping alone is not an escape. Test its actual public movement menu
    without executing a transition, and keep the only-boss fallback when the
    board offers no alternative. Already-entered rewards and ponds are exempt.
    """
    if not fifth_floor_chase_preference_active(state, constraints):
        return actions

    boss_ids = {action.action_id for action in actions if action.kind == ActionKind.MOVE
        and known_fifth_floor_boss(state, action.target_node_id)
        and not public_move_has_development(simulator, state, action, constraints)}
    if not boss_ids:
        return actions
    alternative_move = any(public_move_has_development(simulator, state, action, constraints) for action in actions)
    useful_equips = set()
    for action in actions:
        if action.kind != ActionKind.EQUIP:
            continue
        projected = replace(state, equipped_instance_id=action.equipment_instance_id)
        if any(public_move_has_development(simulator, projected, move, constraints) for move in simulator.legal_actions(projected)):
            useful_equips.add(action.action_id)
    if not alternative_move and not useful_equips:
        return actions
    return tuple(action for action in actions if action.action_id not in boss_ids
        and (action.kind != ActionKind.EQUIP or action.action_id in useful_equips))


def action_is_allowed(simulator, state, action_id, constraints=DEFAULT_POLICY_CONSTRAINTS):
    simulator.decode_action(state, action_id)  # Preserve rejection of an illegal ID.
    return action_id in allowed_action_ids(simulator, state, constraints)


def _preserve_last_redmoss_discard(state, options, actions, constraints):
    """Protect the last preparation tool when another real discard is possible.

    Inventory overflow can leave only negative-value choices. A fixed score
    penalty cannot then guarantee the user's preference; use the actual menu
    instead, while keeping the only legal discard as an escape from overflow.
    Sale, exchange and cultivation rules are unchanged.
    """
    instances = {item.instance_id: item for item in state.item_instances}
    discards = tuple((action, options[action.option_index]) for action in actions
        if action.option_index is not None and (action.kind == ActionKind.DISCARD
            or options[action.option_index].operation == 'discard'))
    protected = {action.action_id for action, option in discards
        if last_redmoss_tool(state, instances.get(option.instance_id), constraints)}
    if protected and any(action.action_id not in protected for action, _ in discards):
        return tuple(action for action in actions if action.action_id not in protected)
    return actions


def _manual_allowed_action_ids(simulator, state, constraints):
    options = simulator.available_options(state)
    original = simulator.legal_actions(state)
    compulsory = compulsory_bank_options(state, options, constraints)
    permitted = tuple(action for action in original
        if (not compulsory or action.option_index is not None and options[action.option_index] in compulsory)
        and (action.option_index is None or option_is_allowed(state, options[action.option_index], constraints))
        and _intentional_employ_move_is_allowed(simulator, state, action, constraints)
        and _intentional_shop_move_is_allowed(simulator, state, action, constraints))
    permitted = _preserve_last_redmoss_discard(state, options, permitted, constraints)
    permitted = _prefer_fifth_floor_chase(simulator, state, permitted, constraints)
    if original and not permitted and not state.terminal:
        raise ConstraintNoLegalAction("the observed menu has no allowed user-policy action (ending, reserve or employment preparation); no decline option or replacement reward was fabricated")
    return tuple(action.action_id for action in permitted)


def _scope_only_action_ids(simulator, state, constraints):
    """Keep every genuine legal action except choices outside the task scope.

    Do not run advisory movement, funding, discard or fifth-floor filters here.
    Affordability and forced/random outcomes remain entirely the simulator's.
    """
    original = simulator.legal_actions(state)
    options = simulator.available_options(state)
    permitted = tuple(action.action_id for action in original
        if action.option_index is None
        or _ending_option_is_allowed(options[action.option_index], constraints))
    if original and not permitted and not state.terminal:
        raise ConstraintNoLegalAction("the observed menu has no action within the first-ending task scope; no replacement option was fabricated")
    return permitted


def allowed_action_ids(simulator, state, constraints=DEFAULT_POLICY_CONSTRAINTS):
    if not constraints.strategy_advice_enabled:
        return _scope_only_action_ids(simulator, state, constraints)
    return _manual_allowed_action_ids(simulator, state, constraints)


def policy_mask_schema(constraints=DEFAULT_POLICY_CONSTRAINTS):
    """Checkpoint identity of the mask actually used by this controller.

    Autonomous checkpoints are independent of manual ranking, route search and
    optional preparation helpers. Game legality has its own environment hash.
    """
    if constraints.strategy_advice_enabled:
        implementation = Path(__file__).read_bytes()
    else:
        implementation = "\n".join(inspect.getsource(function) for function in (
            UserPolicyConstraints.strategy_advice_enabled.fget,
            allowed_action_ids, _scope_only_action_ids, _ending_option_is_allowed,
        )).encode()
        implementation += json.dumps(sorted(FORBIDDEN_ITEM_FLAGS)).encode()
    return {
        "mode": "manual_strategy" if constraints.strategy_advice_enabled else "task_scope_only",
        "configuration": asdict(constraints),
        "implementation_sha256": sha256(implementation).hexdigest(),
    }


class PolicyConstrainedEnvironment:
    """Apply the same user action mask to all search/training objectives."""
    def __init__(self, simulator, constraints=DEFAULT_POLICY_CONSTRAINTS):
        self.simulator = simulator
        self.policy_constraints = constraints

    @property
    def action_size(self):
        return self.simulator.action_size

    def legal_action_ids(self, state):
        return allowed_action_ids(self.simulator, state, self.policy_constraints)

    def transition(self, state, action_id):
        if not action_is_allowed(self.simulator, state, action_id, self.policy_constraints):
            raise ValueError("action violates the user's first-ending or ingot-reserve policy")
        return self.simulator.transition(state, action_id)
