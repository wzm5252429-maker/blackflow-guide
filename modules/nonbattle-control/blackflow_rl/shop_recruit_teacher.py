"""Public menu hypotheses for the isolated shop recruitment mechanism.

No video action is claimed optimal and no future shelf is inspected. The
ticket-only condition values the visible paid recruit's resale margin. The
order condition additionally defers selling corn or discarding a visible
profitable ticket via refresh while that ticket is currently affordable.
"""
from dataclasses import replace

from .bank_entry_variant import BankEntryTeacher
from .domain import NodeType
from .policy_constraints import option_is_allowed


class ShopRecruitTeacher(BankEntryTeacher):
    def __init__(self, simulator, config=None, route_config=None, *, preserve_order=False):
        super().__init__(simulator, config, route_config)
        self.preserve_order = preserve_order

    def _ticket_margin(self, state, option):
        node = state.floor_map.node(state.pending_node_id) if state.pending_node_id else None
        if (not node or node.node_type != NodeType.BATTLE_SHOP
                or not self.simulator.economy.config.full_tech
                or option.operation != 'purchase' or option.item_id != 'service:ticket'):
            return None
        economy = self.simulator.economy
        increase = sum(economy.quote_sell(state, replace(item, appraisal=item.appraisal+3))
                       - economy.quote_sell(state, item)
                       for item in state.item_instances if item.item_id == 'rogue_6_scrap_G_04')
        return increase - max(option.price or 0, -option.effect.gold, 0)

    def _economic_action_score(self, state, action_id):
        state = self.observable_state(state)
        score = super()._economic_action_score(state, action_id)
        action = self.simulator.decode_action(state, action_id)
        if action.option_index is None:
            return score
        options = self.simulator.available_options(state)
        option = options[action.option_index]
        if not option_is_allowed(state, option, self.policy_constraints):
            return score
        margin = self._ticket_margin(state, option)
        if margin is not None:
            return margin * self._gold_value(state) if margin > 0 else min(score, -0.1)
        if self.preserve_order and (option.operation == 'refresh' or (
                option.operation == 'sell' and option.item_id == 'rogue_6_scrap_G_04')):
            profitable = [value * self._gold_value(state) for candidate in options
                if candidate.is_available(state.resources, state.inventory)
                and option_is_allowed(state, candidate, self.policy_constraints)
                and (value := self._ticket_margin(state, candidate)) is not None and value > 0]
            if profitable:
                return min(score, min(profitable)-0.01)
        return score
