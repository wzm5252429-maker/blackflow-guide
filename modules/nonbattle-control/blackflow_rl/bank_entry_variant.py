"""Isolated observed per-entry bank costs; legacy experiment semantics retained.

BV1qBbv6NEcq 05:00–06:15 shows394→331, then331→330→268 at the
same merchant. A second12-withdrawal entry spends63 balance, not84.
The separate simulator fingerprint rejects old checkpoints until explicitly
migrated. Importing this module never changes the legacy simulator.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .domain import EventOption, NodeType, ResourceDelta
from .economy import EconomyEngine, EconomyConfig
from .simulator import BlackflowSimulator
from .route_planner import ObservableRouteEvaluator
from .policy_constraints import preparation_cash_target, option_is_allowed, last_redmoss_tool

EVIDENCE = Path(__file__).resolve().parents[1]/'data/evidence/rogue6_bank_reentry_video_v2.json'


def withdrawal_cost(entry_withdrawn, quantity=1):
    return sum(min(entry_withdrawn+step,7) for step in range(1,quantity+1))


class EntryCostEconomy(EconomyEngine):
    def shop_options(self, state, node):
        # Rebuild only the bank entry; insufficient legacy balance must not
        # hide an affordable corrected quote. Preserve all other options.
        options = [o for o in super().shop_options(state,node) if o.operation != 'bank_withdraw']
        shop = self.shop(state,node.node_id)
        if (node.node_type == NodeType.BATTLE_SHOP and self.config.bank_investment >= 15
                and shop.visits and shop.entry_withdrawn < 12):
            cost = withdrawal_cost(shop.entry_withdrawn)
            if state.bank_balance >= cost:
                option = EventOption('bank_withdraw',f'从银行提取1锭（消耗{cost}余额）',
                    ResourceDelta(gold=1),operation='bank_withdraw',quantity=1,price=cost,ends_node=False)
                leave = next((i for i,o in enumerate(options) if o.operation == 'leave'),len(options))
                options.insert(leave,option)
        return tuple(options)

    def transact(self,state,node,option):
        if option.operation != 'bank_withdraw':
            return super().transact(state,node,option)
        shop = self.shop(state,node.node_id)
        if (node.node_type != NodeType.BATTLE_SHOP or state.pending_node_id != node.node_id
                or self.config.bank_investment < 15 or not shop.visits or shop.entry_withdrawn >= 12):
            raise ValueError('Bank withdrawal requires a live eligible merchant entry')
        cost = withdrawal_cost(shop.entry_withdrawn)
        if state.bank_balance < cost:
            raise ValueError('Insufficient actual bank balance')
        state = replace(state,bank_balance=state.bank_balance-cost,
            bank_balance_spent=state.bank_balance_spent+cost,total_bank_withdrawn=state.total_bank_withdrawn+1)
        shop = replace(shop,entry_withdrawn=shop.entry_withdrawn+1,total_withdrawn=shop.total_withdrawn+1)
        state = self.entry(state,'bank_balance_debit',quantity=cost,
            source=f'bank:node={node.node_id}:visit={shop.visits}:entry_withdrawal={shop.entry_withdrawn}:node_withdrawal={shop.total_withdrawn}:balance={state.bank_balance}')
        state = self.apply(state,ResourceDelta(gold=1),'external_bank_support','bank_withdrawal')
        state = self.save_shop(state,shop)
        return self.set_options(state,node,self.shop_options(state,node))


class BankEntrySimulator(BlackflowSimulator):
    def __init__(self,ruleset=None,map_generator=None,economy_config=None):
        super().__init__(ruleset,map_generator,economy_config)
        self.economy = EntryCostEconomy(economy_config or EconomyConfig())
        self.economy.event_pools = self.ruleset.event_pools

    @property
    def environment_sha256(self):
        return sha256(b'bank-entry-video-variant-v2\0'+super().environment_sha256.encode()
                      +Path(__file__).read_bytes()+EVIDENCE.read_bytes()).hexdigest()


def build(profile=None):
    from .training_environment import TrainingEnvironmentProfile
    legacy = TrainingEnvironmentProfile.load(profile).build()
    return BankEntrySimulator(legacy.ruleset,legacy.map_generator,legacy.economy.config)


class BankEntryTeacher(ObservableRouteEvaluator):
    """Original purchasing preferences with the same corrected funding quote.

    The source task uses multilateral_trade/full_tech: every physical merchant
    entry is a public development opportunity through its real G08 reward.
    Other squads' route-scope bank-only reachability is not validated here.
    """
    def _bank_withdrawal_value(self,state,options,withdrawal):
        if state.floor == 1 and getattr(self.policy_constraints,'first_floor_full_withdrawal',True):
            return 50000.0
        reserve = preparation_cash_target(state,self.policy_constraints)
        if state.resources.gold < self.policy_constraints.minimum_ingot_reserve:
            return 15.0
        shop = next((s for s in state.shops if s.node_id == state.pending_node_id),None)
        used = getattr(shop,'entry_withdrawn',12)
        allowance = max(0,12-used); balance = state.bank_balance; best = -100.0
        goal = self._known_green_departure_cash_target(state)
        if goal is not None and state.resources.gold < goal:
            deficit = goal-state.resources.gold
            sells = [self.action_score(state,self.simulator.ruleset.max_nodes+i)
                for i,candidate in enumerate(options) if candidate.operation == 'sell'
                and candidate.is_available(state.resources,state.inventory)
                and not (candidate.item_id or '').endswith('G_09')
                and not last_redmoss_tool(state,next((item for item in state.item_instances
                    if item.instance_id == candidate.instance_id),None),self.policy_constraints)]
            if max(sells,default=-1.0)>0: return -100.0
            if deficit<=allowance and withdrawal_cost(used,int(deficit))<=balance:
                best = self.config.preparation_cash_shortfall_penalty-.25*(withdrawal.price or 1)
        for index,candidate in enumerate(options):
            if candidate.operation != 'purchase': continue
            cost = candidate.price or max(0,-candidate.effect.gold)
            deficit = cost+reserve-state.resources.gold
            if not 0<deficit<=allowance or withdrawal_cost(used,int(deficit))>balance:
                continue
            funded = replace(state,resources=replace(state.resources,gold=state.resources.gold+deficit))
            if not option_is_allowed(funded,candidate,self.policy_constraints): continue
            value = self.action_score(funded,self.simulator.ruleset.max_nodes+index)
            if value>=8.0:
                best = max(best,min(18.0,value/max(1,deficit))-.25*(withdrawal.price or 1))
        return best
