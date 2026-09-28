"""Research-only ordinary shop recruitment, with no profession sampling.

The old shop layout and price priors remain unchanged. A paid service:ticket
slot now grants the economic action common to all eight ordinary client
tickets: one free reserve recruit, or evidenced retention/decline. Buying,
opening and refreshing never recruit. No ordinary character or promotion
eligibility is invented. This simulator also retains the observed bank fix.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .bank_entry_variant import BankEntrySimulator, EntryCostEconomy
from .domain import EventOption, NodeType
from .operator_economy import recruit_formal
from .recruitment import UnknownReserveOpportunity, eligible_reserve_ids


SOURCE = 'shop_ordinary_common_reserve'
EVIDENCE_TAG = 'VIDEO_PURCHASE_AND_CLIENT_EIGHT_ORDINARY_TICKETS_COMMON_ACTION'
TICKET_IDS = tuple('rogue_6_recruit_ticket_' + profession for profession in
                   ('pioneer', 'warrior', 'tank', 'sniper', 'caster', 'support', 'medic', 'special'))
EVIDENCE = Path(__file__).resolve().parents[1] / 'data/evidence/rogue6_shop_recruit_video_v1.json'


class ShopRecruitEconomy(EntryCostEconomy):
    @staticmethod
    def _shop_opportunities(state):
        return tuple(x for x in state.unknown_recruit_opportunities if x.source == SOURCE)

    def _pending_shop_recruit(self, state):
        return any(x.opportunity_id in state.pending_recruit_ticket_ids
                   for x in self._shop_opportunities(state))

    def recruitment_options(self, state, *, ticket_instance_ids=None):
        options = list(super().recruitment_options(state, ticket_instance_ids=ticket_instance_ids))
        for item in self._shop_opportunities(state):
            nonce = item.opportunity_id
            if ticket_instance_ids is not None and nonce not in ticket_instance_ids:
                continue
            if nonce in state.stored_recruit_ticket_ids and state.floor_map.node(
                    state.current_node_id).node_type != NodeType.FINAL:
                continue
            options.append(EventOption('recruit_shop_reserve:' + nonce,
                '招募预备干员（0希望）', operation='recruit_reserve',
                instance_id=nonce, ends_node=False,
                description='普通职业券共有的免费预备选项；不推断职业或其他干员。'))
        return tuple(options)

    def shop_options(self, state, node):
        # A purchase opens its actual ticket before another shop action.
        if self._pending_shop_recruit(state):
            return self.reward_recruit_options(state)
        return super().shop_options(state, node)

    def _grant_paid_opportunity(self, state, node, slot_id):
        if any(not eligible_reserve_ids(ticket_id) for ticket_id in TICKET_IDS):
            raise ValueError('Every possible ordinary ticket must guarantee this action')
        nonce = 'u' + str(state.unknown_recruit_serial)
        if any(e.operation == 'unknown_recruit_granted' and e.instance_id == nonce
               for e in state.ledger):
            raise ValueError('Shop recruitment nonce was already issued')
        state = replace(state,
            unknown_recruit_opportunities=state.unknown_recruit_opportunities + (
                UnknownReserveOpportunity(nonce, SOURCE),),
            unknown_recruit_serial=state.unknown_recruit_serial + 1,
            pending_recruit_ticket_ids=state.pending_recruit_ticket_ids + (nonce,))
        return self.entry(state, 'unknown_recruit_granted', quantity=1,
            instance_id=nonce, source=f'{EVIDENCE_TAG}:{SOURCE}:{node.node_id}:{slot_id}')

    def transact(self, state, node, option):
        if self._pending_shop_recruit(state):
            raise ValueError('Resolve the purchased ticket before another shop transaction')
        if option.operation != 'purchase' or option.item_id != 'service:ticket':
            return super().transact(state, node, option)
        if (node.node_type != NodeType.BATTLE_SHOP or state.pending_node_id != node.node_id
                or state.current_node_id != node.node_id
                or option not in self.shop_options(state, node)
                or not option.is_available(state.resources, state.inventory)):
            raise ValueError('Ordinary shop recruitment requires one live affordable unsold slot')
        # The original transaction validates/debits the displayed price,
        # increments the legacy acquisition count and marks exactly one slot sold.
        state = super().transact(state, node, option)
        state = self._grant_paid_opportunity(state, node, option.option_id.removeprefix('buy:'))
        return self.set_options(state, node, self.shop_options(state, node))

    def _remove_shop_opportunity(self, state, nonce, operation):
        state = replace(state,
            unknown_recruit_opportunities=tuple(x for x in state.unknown_recruit_opportunities
                                                if x.opportunity_id != nonce),
            pending_recruit_ticket_ids=tuple(x for x in state.pending_recruit_ticket_ids if x != nonce),
            stored_recruit_ticket_ids=tuple(x for x in state.stored_recruit_ticket_ids if x != nonce))
        return self.entry(state, operation, quantity=1, instance_id=nonce,
                          source=f'{EVIDENCE_TAG}:{SOURCE}:{nonce}')

    def after_recruitment_choice(self, state, node):
        if node.node_type == NodeType.BATTLE_SHOP and self._pending_shop_recruit(state):
            return self.set_options(state, node, self.shop_options(state, node))
        return super().after_recruitment_choice(state, node)

    def choose(self, state, node, option):
        own_ids = {x.opportunity_id for x in self._shop_opportunities(state)}
        if option.instance_id not in own_ids:
            # Replayed consumed choices must not fall through to generic helpers.
            if option.option_id.startswith('recruit_shop_reserve:'):
                raise ValueError('Shop recruitment opportunity is absent or already consumed')
            return super().choose(state, node, option)
        if state.chase_reward_context is not None:
            raise ValueError('A stored shop ticket cannot interrupt chase rewards')
        if node.node_id != state.current_node_id:
            raise ValueError('Recruitment must use the current public menu')
        nonce = option.instance_id
        if option.operation == 'recruit_reserve':
            if option not in self.recruitment_options(state):
                raise ValueError('The actual shop ticket is not usable here')
            state = self._remove_shop_opportunity(state, nonce, 'unknown_recruit_consumed')
            state = recruit_formal(self, state, 'observed_reserve:' + nonce,
                                   source=EVIDENCE_TAG + ':' + SOURCE, hope_cost=0)
        elif option.operation in ('retain_recruit_ticket', 'decline_recruitment'):
            if option not in self.reward_recruit_options(state):
                raise ValueError('Shop ticket disposition requires its pending opportunity')
            if option.operation == 'retain_recruit_ticket':
                state = replace(state,
                    pending_recruit_ticket_ids=tuple(x for x in state.pending_recruit_ticket_ids if x != nonce),
                    stored_recruit_ticket_ids=state.stored_recruit_ticket_ids + (nonce,))
                state = self.entry(state, 'recruit_ticket_retained', quantity=1,
                                   instance_id=nonce, source=EVIDENCE_TAG + ':' + SOURCE)
            else:
                state = self._remove_shop_opportunity(state, nonce, 'unknown_recruit_discarded')
        else:
            raise ValueError('Unsupported action on an ordinary shop recruitment opportunity')
        return self.after_recruitment_choice(state, node)


class ShopRecruitSimulator(BankEntrySimulator):
    def __init__(self, ruleset=None, map_generator=None, economy_config=None):
        super().__init__(ruleset, map_generator, economy_config)
        self.economy = ShopRecruitEconomy(self.economy.config)
        self.economy.event_pools = self.ruleset.event_pools

    @property
    def environment_sha256(self):
        return sha256(b'shop-ordinary-recruit-research-v1\0' + super().environment_sha256.encode()
                      + Path(__file__).read_bytes() + EVIDENCE.read_bytes()).hexdigest()


def build(profile=None):
    from .training_environment import TrainingEnvironmentProfile
    legacy = TrainingEnvironmentProfile.load(profile).build()
    return ShopRecruitSimulator(legacy.ruleset, legacy.map_generator, legacy.economy.config)
