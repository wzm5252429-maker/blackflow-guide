"""Isolated public two-action corn-capacity hypothesis; no student integration.

Call choose_action, not just action_score: the visible purchase belongs to the
selected sale/purchase pair. Other policy values and the route search stay old.
Only an already displayed sale is projected, never a refresh or unknown node.
"""
from dataclasses import replace

from .agents import choose_heuristic_action
from .bank_entry_variant import BankEntryTeacher
from .domain import NodeType
from .policy_constraints import allowed_action_ids, option_is_allowed


CORN = 'rogue_6_scrap_G_04'


class CornCapacityTeacher(BankEntryTeacher):
    def __init__(self, simulator, config=None, route_config=None):
        super().__init__(simulator, config, route_config)
        self._pending_corn_purchase = None
        self.last_capacity_decision = None

    @staticmethod
    def _shop_signature(state):
        # Bind the second action to this exact next public shop/asset state.
        # Hidden RNG seeds, future maps and audit histories are not inputs.
        return (state.floor_index, state.floor, state.current_node_id,
                state.pending_node_id, state.step_count, state.resources,
                state.parts_capacity, state.item_instances, state.inventory,
                state.equipped_instance_id, state.shops)

    def choose_action(self, state):
        public = self.observable_state(state)
        self.last_capacity_decision = None
        legal = allowed_action_ids(self.simulator, public, self.policy_constraints)
        options = self.simulator.available_options(public)
        if self._pending_corn_purchase is not None:
            expected, offer, report = self._pending_corn_purchase
            self._pending_corn_purchase = None
            if expected == self._shop_signature(public):
                for index, option in enumerate(options):
                    action = self.simulator.ruleset.max_nodes + index
                    if option == offer and action in legal:
                        self.last_capacity_decision = dict(report, phase='purchase')
                        return action

        baseline = choose_heuristic_action(self, state)
        if (not public.pending_node_id or public.resources.parts != public.parts_capacity
                or public.terminal or not self.config.sell_for_capacity_upgrade):
            return baseline
        node = public.floor_map.node(public.pending_node_id)
        if node.node_type not in (NodeType.SCRAP_SHOP, NodeType.BATTLE_SHOP):
            return baseline
        offers = [o for o in options if o.operation == 'purchase' and o.item_id == CORN
                  and o.is_available(public.resources, public.inventory)
                  and option_is_allowed(public, o, self.policy_constraints)]
        if not offers:
            return baseline
        baseline_score = self.action_score(state, baseline)
        best_score = max(0.0, baseline_score)
        winner = None
        for index, sale in enumerate(options):
            action = self.simulator.ruleset.max_nodes + index
            item = next((x for x in public.item_instances if x.instance_id == sale.instance_id), None)
            if (action not in legal or sale.operation != 'sell' or item is None
                    or item.category not in ('MOVE', 'PASSIVE')):
                continue
            # The actual deterministic transaction handles inventory, equipped
            # gear, and the once-only third-sale bonus. It sees public state.
            after = self.simulator.economy.choose(public, node, sale)
            if after.random_counter != public.random_counter:
                raise ValueError('Capacity sale unexpectedly consumed random outcomes')
            if after.resources.parts >= after.parts_capacity:
                continue
            current = self.simulator.available_options(after)
            after_legal = set(allowed_action_ids(self.simulator, after, self.policy_constraints))
            proceeds = after.resources.gold - public.resources.gold
            for offer in offers:
                match = next(((i, o) for i, o in enumerate(current)
                              if o.option_id == offer.option_id and o.item_id == CORN), None)
                if match is None or self.simulator.ruleset.max_nodes + match[0] not in after_legal:
                    continue
                purchase = match[1]
                cost = max(purchase.price or 0, -purchase.effect.gold, 0)
                pair_score = (self._part_future_value(after, self._definition(CORN))
                              + (proceeds-cost)*self._gold_value(public)
                              - self._held_part_value(public, item) - 1.0)
                if pair_score <= best_score:
                    continue
                best_score = pair_score
                report = {'phase': 'sale', 'sale_instance': item.instance_id,
                          'sale_item': item.item_id, 'purchase_option_id': purchase.option_id,
                          'pair_score': pair_score, 'baseline_action': baseline,
                          'baseline_score': baseline_score,
                          'proceeds': proceeds, 'cost': cost}
                winner = (action, after, purchase, report)
        if winner is None:
            return baseline
        action, after, purchase, report = winner
        expected = replace(after, step_count=public.step_count+1)
        self._pending_corn_purchase = (self._shop_signature(expected), purchase, report)
        self.last_capacity_decision = report
        return action
