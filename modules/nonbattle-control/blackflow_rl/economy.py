"""Item-level economic simulation with explicit, replaceable sampling priors.

Client/written rules define prices, triggers and transaction limits. Unknown
server distributions remain synthetic; none of the priors is a live-game rate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math
import random
from typing import TYPE_CHECKING

from .domain import (Action, ActionKind, BATTLE_TYPES, EventOption, GameState,
                     ItemInstance, LedgerEntry, MapNode, NodeType, ResourceDelta,
                     ResourceState, ShopState, ShopStock, Transition, PendingExpedition)
from . import relic_effects

if TYPE_CHECKING:
    from .simulator import BlackflowSimulator

SHOPS = frozenset((NodeType.BATTLE_SHOP, NodeType.SCRAP_SHOP))
PART_TYPES = frozenset(('MOVE', 'GOODS', 'PASSIVE'))
PREFIX = 'rogue_6_scrap_'


@dataclass(frozen=True, slots=True)
class EconomyConfig:
    squad: str = 'multilateral_trade'
    full_tech: bool = True
    difficulty: int = 15
    bank_investment: int = 500
    # Cumulative investment unlocks are distinct from the spendable balance.
    initial_bank_balance: int = 999
    observed_initial_bank_balance: int | None = None
    random_transport_unknown_first: bool = True
    # These probabilities are research priors, NOT measured game drop rates.
    normal_relic_probability: float = 0.15
    battle_part_probability: float = 0.8
    seed_part_probability: float = 0.8
    stronghold_relic_probability: float = 1.0
    combat_assumption: str = 'guaranteed_victory'
    assume_all_chests_opened: bool = True
    # Account/run observations are separate from full technology and bank level.
    previous_cleared_floors: int | None = 2
    observed_starting_choice_ids: tuple[str, ...] = ()
    first_ending_unlocked: bool | None = True
    mechanist_full_mastery: bool = True
    default_initial_mechanist: bool = True
    initial_formal_operator_ids: tuple[str, ...] = ()
    # The default choice is a full-mastery E1 Mechanist (2 hope) and two
    # anonymous free formal companions. Actual IDs/cost observations override it.
    abstract_initial_formal_operator_count: int = 3
    initial_recruitment_hope_spent: int | None = None
    # Override candidate counts only when an actual battle offer was observed.
    observed_battle_recruit_candidate_counts: tuple[tuple[str, int], ...] = ()
    enabled: bool = True
    # Append fields: historical frozen/slotted snapshots used positional values.
    # Sensitivity parameters never override catalog-confirmed prices.
    synthetic_relic_price_multiplier: float = 1.0
    synthetic_battle_shop_base_relic_slots: int = 2
    # None preserves observation-only behavior. A number explicitly enables
    # the video-inspired integer wave prior; it is not a server probability.
    synthetic_wave_tail_probability: float | None = None

    def __getstate__(self):
        from dataclasses import asdict
        return {'version': 1, 'values': asdict(self)}

    def __setstate__(self, state):
        from dataclasses import fields
        definitions=fields(self)
        if isinstance(state, dict):
            if state.get('version') != 1:
                raise ValueError('unsupported economy configuration snapshot')
            values=state['values']
            if set(values)-{field.name for field in definitions}:
                raise ValueError('unknown economy configuration snapshot fields')
        else:
            # Legacy dataclass-generated __getstate__ serialized a value list.
            if len(state)>len(definitions):
                raise ValueError('economy configuration snapshot has too many fields')
            values=dict(zip((field.name for field in definitions),state))
        for definition in definitions:
            object.__setattr__(self,definition.name,values.get(definition.name,definition.default))
        self.__post_init__()

    def __post_init__(self):
        from .wave_model import validate_tail_probability
        validate_tail_probability(self.synthetic_wave_tail_probability)
        if (type(self.synthetic_relic_price_multiplier) not in (int, float)
                or not math.isfinite(self.synthetic_relic_price_multiplier)
                or self.synthetic_relic_price_multiplier <= 0):
            raise ValueError('synthetic relic price multiplier must be finite and positive')
        if (type(self.synthetic_battle_shop_base_relic_slots) is not int
                or not 0 <= self.synthetic_battle_shop_base_relic_slots <= 12):
            raise ValueError('synthetic base relic slots must be an integer in 0..12')
        if self.squad not in ('multilateral_trade', 'default'):
            raise ValueError('unsupported squad')
        if self.combat_assumption != 'guaranteed_victory':
            raise ValueError('only guaranteed_victory is implemented; observed battles require actual outcome ingestion')
        if not 0 <= self.difficulty <= 15:
            raise ValueError('difficulty must be 0..15')
        if type(self.initial_bank_balance) is not int:
            raise ValueError('initial bank balance must be a concrete integer assumption')
        for balance in (self.initial_bank_balance,self.observed_initial_bank_balance):
            if balance is not None and (type(balance) is not int or not 0<=balance<=999):
                raise ValueError('initial bank balance must be an integer in 0..999')
        if self.previous_cleared_floors is not None and self.previous_cleared_floors < 0:
            raise ValueError('previous_cleared_floors cannot be negative')
        if not 0 <= self.abstract_initial_formal_operator_count <= 3:
            raise ValueError('abstract initial formal operator count must be 0..3')
        if len(set(self.initial_formal_operator_ids)) != len(self.initial_formal_operator_ids):
            raise ValueError('initial formal operator IDs must be distinct')
        if any(not x for x in self.initial_formal_operator_ids):
            raise ValueError('initial formal operator IDs cannot be empty')
        if self.initial_recruitment_hope_spent is not None and (
                type(self.initial_recruitment_hope_spent) is not int or self.initial_recruitment_hope_spent < 0):
            raise ValueError('initial recruitment hope spent must be a nonnegative integer')
        if (self.initial_recruitment_hope_spent is None and any(
                operator_id != 'char_4230_mcnist' for operator_id in self.initial_formal_operator_ids)):
            # Arbitrary explicit identities are not the default free-companion
            # abstraction. Their observed initial menu can include special
            # discounts; never silently turn a named five/six-star into a free
            # recruit when that total quote was not supplied.
            raise ValueError('explicit non-Mechanist initial operators require observed initial_recruitment_hope_spent')
        counts=self.observed_battle_recruit_candidate_counts
        if len({kind for kind,count in counts})!=len(counts) or any(
                kind not in ('BATTLE_NORMAL','BATTLE_ELITE','BATTLE_BOSS','BOSS_CHASE')
                or type(count) is not int or count<1 for kind,count in counts):
            raise ValueError('observed battle candidate counts require distinct battle types and positive integers')
        if self.observed_starting_choice_ids and (
                self.previous_cleared_floors is None or self.previous_cleared_floors < 2):
            raise ValueError('observed baseline starting offer requires at least two previous cleared floors')
        for x in (self.normal_relic_probability, self.battle_part_probability,
                  self.seed_part_probability,self.stronghold_relic_probability):
            if not 0 <= x <= 1:
                raise ValueError('synthetic probabilities must be in [0,1]')


class EconomyEngine:
    def __init__(self, config: EconomyConfig):
        from .catalog import load_catalog
        self.config = config
        self.catalog = load_catalog()

    def counter(self, state, key):
        return dict(state.event_counters).get(key, 0)

    def set_counter(self, state, key, value):
        counters = dict(state.event_counters)
        counters[key] = int(value)
        return replace(state, event_counters=tuple(sorted(counters.items())))

    def rng(self, state, purpose):
        digest = sha256(f'{state.economy_seed}:{state.random_counter}:{purpose}'.encode()).digest()
        return random.Random(int.from_bytes(digest[:8], 'big'))

    def roll(self, state, purpose):
        return self.rng(state, purpose), replace(state, random_counter=state.random_counter + 1)

    def entry(self, state, operation, *, item=None, instance_id=None, quantity=0, gold=0, source=''):
        return replace(state, ledger=state.ledger + (LedgerEntry(
            state.step_count + 1, state.floor, state.current_node_id, operation,
            item.item_id if item else None, item.instance_id if item else instance_id,
            item.category if item else None, quantity, gold, source),))

    def apply(self, state, effect, source, operation='reward'):
        if any(getattr(effect, x) for x in ('parts', 'relics')):
            raise ValueError('item counts must come from item instances')
        before = state.resources
        after = before.apply(effect)
        if before.gold + effect.gold < 0:
            raise ValueError('insufficient ingots')
        state = replace(state, resources=after)
        if after.gold != before.gold:
            state = self.entry(state, operation, gold=after.gold - before.gold, source=source)
        return state

    def sync(self, state):
        return replace(state, resources=replace(state.resources,
            relics=sum(x.category == 'RELIC' for x in state.item_instances),
            parts=sum(x.category in PART_TYPES for x in state.item_instances)),
            inventory=frozenset(x.item_id for x in state.item_instances if x.category == 'RELIC'))

    def acquire(self, state, item_id, source):
        definition = self.catalog.variant(item_id, difficulty=self.config.difficulty)
        item_id = definition.item_id
        if definition.category == 'RELIC' and any(
                self.catalog.items[x.item_id].canonical_id == definition.canonical_id
                for x in state.item_instances if x.category == 'RELIC'):
            return state  # unique collectible: never bill or count a duplicate
        instance = ItemInstance(f'i{state.item_serial}', item_id, definition.category,
            definition.move_uses if definition.category == 'MOVE' else definition.passive_uses,
            int(definition.sell_price or 0))
        state = replace(state, item_instances=state.item_instances + (instance,), item_serial=state.item_serial + 1)
        state = self.entry(state, 'acquire', item=instance, quantity=1, source=source)
        state = self.sync(state)
        if definition.category in PART_TYPES:
            state = relic_effects.on_scrap_acquired(self, state, instance)
        if item_id == PREFIX+'G_07':
            # The three balls are three real acquisitions, including valuation triggers.
            for _ in range(3):
                state = self.acquire(state, PREFIX+'G_08', 'multimoss')
        if definition.category == 'RELIC':
            state = self.relic_immediate(state, definition)
        if item_id=='rogue_6_relic_cargo_14':
            from .notebook_observation import on_notebook_acquired
            state=self.apply_notebook_update(state,on_notebook_acquired(source=source))
        return state

    def remove(self, state, instance_id, operation, source):
        item = next(x for x in state.item_instances if x.instance_id == instance_id)
        state = replace(state, item_instances=tuple(x for x in state.item_instances if x.instance_id != instance_id),
                        equipped_instance_id=None if state.equipped_instance_id == instance_id else state.equipped_instance_id,
                        pending_recruit_ticket_ids=tuple(x for x in state.pending_recruit_ticket_ids if x!=instance_id),
                        stored_recruit_ticket_ids=tuple(x for x in state.stored_recruit_ticket_ids if x!=instance_id),
                        temporary_recruit_offers=tuple(x for x in state.temporary_recruit_offers
                                                     if x.ticket_instance_id!=instance_id))
        state = self.entry(state, operation, item=item, quantity=1, source=source)
        if item.item_id=='rogue_6_relic_cargo_14' and state.notebook_observation is not None:
            state=replace(state,notebook_observation=replace(state.notebook_observation,
                held=False,observed_layer=None,layer_source=None))
        return self.sync(state)

    def relic_immediate(self, state, definition):
        state = relic_effects.apply_immediate(self, state, definition)
        return relic_effects.on_commander_level(self, state, max(1,self.counter(state,'commander_level')))

    def apply_notebook_update(self,state,update):
        # Validate every observed reward before recording the observation or
        # granting anything. Pool names, scraps and duplicate variants are not
        # observations of distinct extra collectible drops.
        canonical=[]
        for item_id in update.grant_item_ids:
            definition=self.catalog.items.get(item_id)
            if definition is None or definition.category!='RELIC':
                raise ValueError('observed notebook rewards must be concrete collectible IDs')
            canonical.append(definition.canonical_id)
        owned={self.catalog.items[item.item_id].canonical_id for item in state.item_instances if item.category=='RELIC'}
        if len(set(canonical))!=len(canonical) or any(key in owned for key in canonical):
            raise ValueError('observed notebook extra reward duplicates a collectible identity')
        state=replace(state,notebook_observation=update.state)
        for audit in update.audit:
            source=audit.source+':layer='+str(audit.layer if audit.layer is not None else 'unknown')
            if audit.battle_id is not None:
                source+=':battle_id='+audit.battle_id
            state=self.entry(state,audit.operation,source=source)
        for item_id in update.grant_item_ids:
            state=self.acquire(state,item_id,'observed_notebook_extra_drop')
        return state

    def observe_notebook_layer(self,state,layer,*,source):
        from .notebook_observation import observe_notebook_layer,on_notebook_acquired,NOTEBOOK_ID
        if NOTEBOOK_ID not in state.inventory:
            raise ValueError('cannot observe an unheld notebook')
        if state.notebook_observation is None:
            state=self.apply_notebook_update(state,on_notebook_acquired(source='provided_notebook_inventory'))
        return self.apply_notebook_update(state,observe_notebook_layer(state.notebook_observation,layer,source=source))

    def observe_temporary_recruitment(self,state,*,ticket_instance_id,observed_char_id,observation_source):
        from .observed_operator_bridge import observe_temporary_recruitment
        return observe_temporary_recruitment(self,state,ticket_instance_id=ticket_instance_id,
            observed_char_id=observed_char_id,observation_source=observation_source)

    def observe_employment_menu(self,state,*,observed_candidates,observation_source):
        from .observed_operator_bridge import observe_employment_menu
        return observe_employment_menu(self,state,observed_candidates=observed_candidates,
                                       observation_source=observation_source)

    def observe_employment_refresh(self,state,*,observed_candidates,observation_source):
        from .observed_operator_bridge import observe_employment_refresh
        return observe_employment_refresh(self,state,observed_candidates=observed_candidates,
                                          observation_source=observation_source)

    def observe_emergency_battle_participants(self,state,*,battle_number,participating_emergency_ids):
        from .observed_operator_bridge import observe_emergency_battle_participants
        return observe_emergency_battle_participants(self,state,battle_number=battle_number,
            participating_emergency_ids=participating_emergency_ids)

    def notebook_after_battle(self,state,*,battle_id,stage_id=None,observed_extra_drop=None):
        from .notebook_observation import after_battle,on_notebook_acquired,NOTEBOOK_ID
        if NOTEBOOK_ID not in state.inventory:
            if observed_extra_drop is not None:
                raise ValueError('notebook extra drops require actual notebook ownership')
            return state
        if state.notebook_observation is None:
            state=self.apply_notebook_update(state,on_notebook_acquired(source='provided_notebook_inventory'))
        return self.apply_notebook_update(state,after_battle(state.notebook_observation,
            battle_id=battle_id,stage_id=stage_id,observed_extra_drop=observed_extra_drop))

    def audit_notebook_battle(self,state,*,stage_id=None):
        if 'rogue_6_relic_cargo_14' not in state.inventory:
            return state
        serial=self.counter(state,'notebook:battles_seen')+1
        state=self.set_counter(state,'notebook:battles_seen',serial)
        return self.notebook_after_battle(state,battle_id='notebook_battle:'+str(serial),stage_id=stage_id)

    def candidates(self, state, category, *, shop=False, rarity=None, exclude=()):
        from .economic_sampling import candidate_ids
        candidates=candidate_ids(self.catalog, category, difficulty=self.config.difficulty,
            owned=(x.item_id for x in state.item_instances if x.category == 'RELIC'),
            exclude=exclude, shop=shop, rarity=rarity)
        if shop and (state.floor not in (1,2,3) or not self.first_ending_available()):
            candidates=[x for x in candidates if x!='rogue_6_relic_final_2']
        return candidates

    def grant(self, state, item_id, quantity=1, source='node'):
        from .economic_sampling import choose_from_pool, source_pool, source_pool_assumption
        for _ in range(quantity):
            if item_id.startswith('pool:'):
                category = item_id.removeprefix('pool:')
                rarity = None
                if category not in self.catalog.observed_pools and ':' in category:
                    category, rarity_name = category.split(':',1)
                    rarity = (rarity_name,)
                pool_id = category if category in self.catalog.observed_pools else source_pool(category, source)
                if category not in self.catalog.observed_pools:
                    assumption = source_pool_assumption(category, source, pool_id)
                    if assumption:
                        state = self.entry(state, 'model_assumption', source=assumption)
                rng, state = self.roll(state, 'pool:' + category + ':' + source)
                if category in self.catalog.observed_pools:
                    kinds = ('RELIC', 'MOVE', 'GOODS', 'PASSIVE')
                    candidates = [x for kind in kinds for x in self.candidates(state, kind)]
                else:
                    candidates = self.candidates(state, category,rarity=rarity)
                selected, sampling = choose_from_pool(self.catalog, rng, candidates,
                    pool_id=pool_id, difficulty=self.config.difficulty)
                if selected is None:
                    state = self.entry(state, 'pool_exhausted', source=pool_id or category)
                    continue
                state = self.entry(state, 'sampling', source=(pool_id or category) + ':' + sampling)
            else:
                selected = self.catalog.variant(item_id, difficulty=self.config.difficulty).item_id
            state = self.acquire(state, selected, source)
        return state

    def quote_sell(self, state, instance):
        from .utopia_economy import shop_terms
        region=state.region_state
        ideology=region.ideology if region and region.affects(state.current_node_id) else None
        multiplier=shop_terms(ideology,self.config.difficulty).sell_multiplier
        return max(0,math.floor(instance.appraisal*multiplier))

    def quote_buy(self, state, item, *, base_price=None):
        from .economic_sampling import undiscounted_price
        definition = self.catalog.items[item] if isinstance(item,str) else item
        price = self.base_shop_price(definition.item_id)[0] if base_price is None else base_price
        if 'rogue_6_relic_legacy_111' in state.inventory:
            price = math.ceil(price * 0.5)
        return max(0,int(price))

    def base_shop_price(self, item_id):
        from .economic_sampling import undiscounted_price
        price, evidence = undiscounted_price(self.catalog, item_id)
        if evidence.startswith('synthetic') and self.config.synthetic_relic_price_multiplier != 1:
            price = math.ceil(price * self.config.synthetic_relic_price_multiplier)
            evidence += ':sensitivity_multiplier=' + str(self.config.synthetic_relic_price_multiplier)
        return price, evidence

    def battle_gold(self,state,base_gold,source):
        from fractions import Fraction
        multiplier=Fraction(1)
        modifiers=relic_effects.reward_up_parameters(self,state)
        for _,amount in modifiers:
            multiplier*=1+Fraction(str(amount))
        if modifiers:
            state=self.entry(state,'model_assumption',source='battle_reward:stack_multiply_round_down')
        return self.apply(state,ResourceDelta(gold=int(base_gold*multiplier)),source)

    def shop(self, state, node_id):
        return next((x for x in state.shops if x.node_id == node_id), ShopState(node_id))

    def save_shop(self, state, shop):
        return replace(state, shops=tuple(x for x in state.shops if x.node_id != shop.node_id)+(shop,))

    def stock_shop(self, state, node, shop):
        from .economic_sampling import (SYNTHETIC_SHOP_LAYOUT, battle_shop_categories,
            choose_from_pool, undiscounted_price)
        rng, state = self.roll(state, 'stock:'+node.node_type.value)
        if node.node_type == NodeType.BATTLE_SHOP:
            categories = battle_shop_categories(difficulty=self.config.difficulty,
                full_technology=self.config.full_tech, bank_investment=self.config.bank_investment,
                base_relic_slots=self.config.synthetic_battle_shop_base_relic_slots)
        else:
            categories = SYNTHETIC_SHOP_LAYOUT['scrap_shop_categories']
        from .utopia_economy import shop_terms
        region=state.region_state
        ideology=region.ideology if region and region.affects(node.node_id) else None
        loss=max(shop.lost_slots,shop_terms(ideology,self.config.difficulty).slot_reduction)
        if loss:
            # Which base slots vanish is not exposed by the public source.
            indices=sorted(rng.sample(range(len(categories)),max(0,len(categories)-loss)))
            categories=tuple(categories[i] for i in indices)
            shop=replace(shop,lost_slots=loss)
            state=self.entry(state,'model_assumption',source='storage_room:uniform_removed_slots_round_down_sale')
        state = self.entry(state, 'model_assumption', source='shop:unverified_base_slot_layout')
        stock, offered = [], []
        for index, category in enumerate(categories):
            if category in ('TICKET','TOOL'):
                item_id, price = ('service:ticket',4) if category == 'TICKET' else ('service:tool',3)
                state = self.entry(state, 'model_assumption', source='shop:service_price_prior')
            else:
                actual_category = 'RELIC' if category == 'TRAINING_RELIC' else category
                candidates = self.candidates(state, actual_category, shop=True,
                    exclude=offered if actual_category == 'RELIC' else ())
                training_ids = {f'rogue_6_relic_assign_{i}' for i in range(1,9)}
                if category == 'TRAINING_RELIC':
                    candidates = [x for x in candidates if x in training_ids]
                elif category == 'RELIC':
                    candidates = [x for x in candidates if x not in training_ids]
                if node.node_type == NodeType.BATTLE_SHOP:
                    subpool = 'new-training' if category == 'TRAINING_RELIC' else 'other-collectibles' if category == 'RELIC' else 'parts'
                    item_id, sampling = choose_from_pool(self.catalog, rng, candidates,
                        pool_id='rogue_6:诡意行商', subpool=subpool, difficulty=self.config.difficulty)
                else:
                    item_id, sampling = choose_from_pool(self.catalog, rng, candidates,
                        pool_id=None, difficulty=self.config.difficulty)
                if item_id is None:
                    continue
                price, price_evidence = self.base_shop_price(item_id)
                state = self.entry(state, 'sampling', source='shop:' + sampling)
                if price_evidence.startswith('synthetic'):
                    state = self.entry(state, 'model_assumption', source='shop:' + price_evidence)
                offered.append(item_id)
            stock.append(ShopStock(f'{shop.refreshes}:{index}', item_id, int(price)))
        return state, replace(shop, stock=tuple(stock))

    def shop_options(self, state, node):
        shop = self.shop(state,node.node_id)
        options = []
        for slot in shop.stock:
            if slot.sold:
                continue
            if slot.item_id.startswith('service:'):
                name, price = ('招募券' if slot.item_id.endswith('ticket') else '战术道具'), slot.price
                if 'rogue_6_relic_legacy_111' in state.inventory:
                    price=math.ceil(price*0.5)
                effect = ResourceDelta(gold=-price, tickets=int(slot.item_id.endswith('ticket')))
            else:
                definition = self.catalog.items[slot.item_id]
                if definition.category == 'RELIC' and slot.item_id in state.inventory:
                    continue
                name, price = definition.name, self.quote_buy(state,definition,base_price=slot.price)
                effect = ResourceDelta(gold=-price)
            options.append(EventOption('buy:'+slot.slot_id, f'{price}锭购买{name}',effect,
                operation='purchase',item_id=slot.item_id,price=price,ends_node=False))
        for item in state.item_instances:
            if item.category in PART_TYPES and (node.node_type == NodeType.SCRAP_SHOP or self.config.full_tech):
                price = self.quote_sell(state,item)
                options.append(EventOption('sell:'+item.instance_id,
                    f'出售{self.catalog.items[item.item_id].name}（{price}锭）',ResourceDelta(gold=price),
                    operation='sell',item_id=item.item_id,instance_id=item.instance_id,price=price,ends_node=False))
        if self.config.full_tech and shop.refreshes < 4:
            price = 4*(shop.refreshes+1)
            options.append(EventOption('refresh',f'{price}锭刷新商品',ResourceDelta(gold=-price),
                operation='refresh',price=price,ends_node=False))
        if node.node_type == NodeType.SCRAP_SHOP:
            for item in state.item_instances:
                if item.item_id == PREFIX+'G_01':
                    options.append(EventOption('cultivate:'+item.instance_id,'培育种子',
                        operation='cultivate',item_id=item.item_id,instance_id=item.instance_id,ends_node=False))
        if node.node_type==NodeType.BATTLE_SHOP and self.config.bank_investment>=15 and shop.visits and shop.entry_withdrawn<12:
            bank_cost=min(shop.total_withdrawn+1,7)
            if state.bank_balance>=bank_cost:
                options.append(EventOption('bank_withdraw','从银行提取1锭（消耗'+str(bank_cost)+'余额）',
                    ResourceDelta(gold=1),operation='bank_withdraw',quantity=1,
                    price=bank_cost,ends_node=False))
        options.append(EventOption('leave','离开行商',operation='leave'))
        return tuple(options)

    def recruitment_options(self,state,*,ticket_instance_ids=None):
        from .recruitment import reserve_recruit_options,unknown_reserve_recruit_options
        eligible=frozenset(item.instance_id for item in state.item_instances if item.category=='RECRUIT_TICKET')
        unknown_eligible=frozenset(item.opportunity_id for item in state.unknown_recruit_opportunities)
        if state.floor_map.node(state.current_node_id).node_type!=NodeType.FINAL:
            eligible-=frozenset(state.stored_recruit_ticket_ids)
            unknown_eligible-=frozenset(state.stored_recruit_ticket_ids)
        if ticket_instance_ids is not None:
            eligible&=frozenset(ticket_instance_ids)
            unknown_eligible&=frozenset(ticket_instance_ids)
        result=list(reserve_recruit_options(state,ticket_instance_ids=eligible))
        result.extend(unknown_reserve_recruit_options(state,ticket_instance_ids=unknown_eligible))
        from .temporary_recruitment import (TEMPORARY_TICKET_IDS,temporary_recruit_options,
                                            load_pinned_temporary_candidate_catalog)
        if state.temporary_recruit_offers:
            from .observed_operator_bridge import emergency_character_ids
            result.extend(temporary_recruit_options(self,state,load_pinned_temporary_candidate_catalog(),
                tuple(x for x in state.temporary_recruit_offers if x.ticket_instance_id in eligible),
                blocked_operator_ids=emergency_character_ids(state)))
        from .operator_economy import MECHANIST_ID,mechanist_eligible_ticket_ids,mechanist_hope_cost
        if MECHANIST_ID in state.available_formal_operator_ids and MECHANIST_ID not in state.promoted_operator_ids:
            hope=mechanist_hope_cost(promotion=True,full_mastery=self.config.mechanist_full_mastery)
            if state.resources.hope>=hope:
                for item in state.item_instances:
                    if (item.instance_id in eligible and item.item_id in mechanist_eligible_ticket_ids()
                            and not item.item_id.startswith('rogue_6_recruit_ticket_temp_')):
                        result.append(EventOption('mechanist_promote:'+item.instance_id,
                            f'使用{self.catalog[item.item_id].name}进阶机械师（{hope}希望）',
                            ResourceDelta(hope=-hope),operation='mechanist_promote',
                            item_id=MECHANIST_ID,instance_id=item.instance_id,ends_node=False))
        from .upgrade_ticket_actions import mechanist_upgrade_ticket_options
        result.extend(option for option in mechanist_upgrade_ticket_options(state)
            if ticket_instance_ids is None or option.instance_id in ticket_instance_ids)
        return tuple(result)

    def reward_recruit_options(self,state):
        from .recruitment_candidates import recruit_candidate_options
        pending=frozenset(state.pending_recruit_ticket_ids)
        result=list(self.recruitment_options(state,ticket_instance_ids=pending))
        for group in state.pending_recruit_candidate_groups:
            result.extend(recruit_candidate_options(group,name_of=lambda item_id:self.catalog[item_id].name))
        held_ids=tuple(item.instance_id for item in state.item_instances)+tuple(
            item.opportunity_id for item in state.unknown_recruit_opportunities)
        for instance_id in held_ids:
            if instance_id not in pending:
                continue
            if len(state.stored_recruit_ticket_ids)<3:
                result.append(EventOption('retain_recruit_ticket:'+instance_id,'留存此招募券（险路尽头可取出）',
                    operation='retain_recruit_ticket',instance_id=instance_id,ends_node=False))
            result.append(EventOption('decline_recruitment:'+instance_id,'放弃此招募券的招募',
                operation='decline_recruitment',instance_id=instance_id,ends_node=False))
        return tuple(result)

    def after_recruitment_choice(self,state,node):
        if state.chase_reward_context is not None:
            return self.chase_reward_menu(state)
        if state.pending_node_id is None:
            return state
        if state.chase_reward_pending:
            recruits=self.reward_recruit_options(state)
            return self.set_options(state,node,recruits) if recruits else replace(state,pending_node_id=None)
        if node.node_type in SHOPS:
            return self.set_options(state,node,self.shop_options(state,node)+self.reward_recruit_options(state))
        if node.node_type==NodeType.FINAL:
            return self.set_options(state,node,tuple(x for x in node.options
                if x.operation not in ('recruit_reserve','mechanist_promote','recruit_temporary',
                                       'retain_recruit_ticket','decline_recruitment'))+
                self.recruitment_options(state,ticket_instance_ids=state.stored_recruit_ticket_ids)+
                self.reward_recruit_options(state))
        # Recruitment is a separate decision alongside already rolled loot.
        # Preserve the existing take alternatives, never reroll them for free.
        takes=tuple(x for x in node.options if x.operation=='take')
        if takes:
            return self.set_options(state,node,takes+self.reward_recruit_options(state))
        return self.reward_menu(state,node)

    def navigation_options(self, state):
        result=list(self.recruitment_options(state))
        if state.portal_context is not None:
            result.append(EventOption('portal_return','结束黑潭探索，返回入口',operation='portal_return'))
        if state.equipped_instance_id:
            result.append(EventOption('unequip','切换徒步跋涉',operation='equip',ends_node=False))
        for item in state.item_instances:
            if item.category == 'MOVE' and item.instance_id != state.equipped_instance_id:
                result.append(EventOption('equip:'+item.instance_id,
                    f'装载{self.catalog.items[item.item_id].name}（余{item.uses_remaining}次）',
                    operation='equip',item_id=item.item_id,instance_id=item.instance_id,ends_node=False))
        if state.resources.parts > state.parts_capacity:
            result = []
            for item in state.item_instances:
                if item.category in PART_TYPES:
                    result.append(EventOption('discard:'+item.instance_id,
                        f'丢弃{self.catalog.items[item.item_id].name}', operation='discard',
                        item_id=item.item_id,instance_id=item.instance_id,ends_node=False))
        return tuple(result)

    def set_options(self, state, node, options, *, observation=False):
        if not options and not observation:
            raise ValueError('empty interaction must be resolved explicitly')
        updated = replace(node, options=tuple(options), auto_effect=ResourceDelta(),requires_observation=observation)
        floor_map = replace(state.floor_map, nodes=tuple(updated if x.node_id == node.node_id else x for x in state.floor_map.nodes))
        maps = list(state.maps)
        maps[state.floor_index] = floor_map
        return replace(state,maps=tuple(maps),pending_node_id=node.node_id)

    def finish(self, state, node):
        if state.chase_reward_context is not None:
            raise ValueError('off-map chase rewards cannot complete a map node')
        if (node.node_type==NodeType.EMPLOY and state.employment_context is not None
                and state.employment_context.active_visit is not None):
            from .employment import end_employment_visit
            state=replace(state,employment_context=end_employment_visit(state.employment_context,
                paid_refresh_cost=self.counter(state,node.node_id+':employment_refresh_paid')))
            state=self.set_counter(state,node.node_id+':employment_refresh_paid',0)
        if self.counter(state,node.node_id+':resident_battle'):
            from .residents import resolve_resident_victory
            floor_map,context=resolve_resident_victory(state.floor_map,state.resident_context,node.node_id)
            maps=list(state.maps)
            maps[state.floor_index]=floor_map
            state=replace(state,maps=tuple(maps),resident_context=context)
            state=self.set_counter(state,node.node_id+':resident_battle',0)
        if node.node_id not in state.completed:
            state = relic_effects.on_node_complete(self,state,node)
            if state.portal_context is not None:
                from .portal import portal_completion_reward
                context,effect=portal_completion_reward(state.portal_context,node)
                state=replace(state,portal_context=context)
                state=self.apply(state,effect,'portal:completion')
            if state.region_state is not None:
                state=replace(state,region_state=state.region_state.clear_source(node.node_id))
        # Finished ordinary events/battles become empty ground. Their old type
        # must not keep activating painted concepts on later journeys.
        if not node.is_exit and node.node_type not in SHOPS | {NodeType.START,NodeType.DOOR,NodeType.LIGHT} and not (
                node.node_type==NodeType.STORY and node.repeatable):
            updated=replace(node,node_type=NodeType.EMPTY,options=(),auto_effect=ResourceDelta())
            maps=list(state.maps)
            maps[state.floor_index]=replace(state.floor_map,nodes=tuple(updated if x.node_id==node.node_id else x for x in state.floor_map.nodes),fingerprint='')
            state=replace(state,maps=tuple(maps))
        from .events import record_incident_seen
        state=record_incident_seen(state,node.event_name) if node.event_name else state
        return replace(state, completed=state.completed | {node.node_id},pending_node_id=None)

    def enter_shop(self, state, node):
        shop = self.shop(state,node.node_id)
        # Reopening the same interaction is not a new physical visit. A legal
        # repeat entry pays/consumes movement and increments movement_count.
        if shop.visits and shop.last_entry_movement_count==state.movement_count:
            return self.set_options(state,node,self.shop_options(state,node))
        if not shop.visits:
            state, shop = self.stock_shop(state,node,shop)
        shop = replace(shop,visits=shop.visits+1,entry_withdrawn=0,
            last_entry_movement_count=state.movement_count)
        state = self.save_shop(state,shop)
        if node.node_type==NodeType.BATTLE_SHOP:
            state=self.entry(state,'bank_entry_allowance',quantity=12,
                source='USER_OBSERVED_RULE:12_ingots_per_actual_shop_entry:visit='+str(shop.visits))
        if self.config.squad == 'multilateral_trade' and self.config.full_tech:
            state = self.acquire(state,PREFIX+'G_08','multilateral_trade:enter_shop')
        return self.set_options(state,node,self.shop_options(state,node))

    def transact(self, state, node, option):
        shop = self.shop(state,node.node_id)
        op = option.operation
        if op=='bank_withdraw':
            if (node.node_type!=NodeType.BATTLE_SHOP or state.pending_node_id!=node.node_id
                    or self.config.bank_investment<15 or not shop.visits or shop.entry_withdrawn>=12):
                raise ValueError('bank withdrawal requires an open green-shop entry with remaining allowance')
            bank_cost=min(shop.total_withdrawn+1,7)
            if state.bank_balance<bank_cost:
                raise ValueError('insufficient actual bank balance')
            state=replace(state,bank_balance=state.bank_balance-bank_cost,
                bank_balance_spent=state.bank_balance_spent+bank_cost,
                total_bank_withdrawn=state.total_bank_withdrawn+1)
            shop=replace(shop,entry_withdrawn=shop.entry_withdrawn+1,total_withdrawn=shop.total_withdrawn+1)
            state=self.entry(state,'bank_balance_debit',quantity=bank_cost,
                source=f'bank:node={node.node_id}:visit={shop.visits}:entry_withdrawal={shop.entry_withdrawn}:node_withdrawal={shop.total_withdrawn}:balance={state.bank_balance}')
            state=self.apply(state,ResourceDelta(gold=1),'external_bank_support','bank_withdrawal')
        elif op == 'purchase':
            slot_id = option.option_id.removeprefix('buy:')
            slot = next(x for x in shop.stock if x.slot_id == slot_id and not x.sold)
            if slot.item_id in state.inventory:
                raise ValueError('unique item already owned')
            state = self.apply(state,option.effect,'shop','purchase')
            shop = replace(shop,stock=tuple(replace(x,sold=True) if x.slot_id==slot_id else x for x in shop.stock))
            if not option.item_id.startswith('service:'):
                state = self.acquire(state,option.item_id,'shop')
        elif op == 'sell':
            item = next(x for x in state.item_instances if x.instance_id == option.instance_id)
            price = self.quote_sell(state,item)
            state = self.remove(state,item.instance_id,'sell','shop')
            state = self.apply(state,ResourceDelta(gold=price),'shop_sale')
            shop = replace(shop,sold_parts=shop.sold_parts+1)
            if self.config.squad == 'multilateral_trade' and shop.sold_parts >= 3 and not shop.trade_bonus_paid:
                state = self.apply(state,ResourceDelta(gold=8),'multilateral_trade:three_sales')
                shop = replace(shop,trade_bonus_paid=True)
        elif op == 'refresh':
            if not self.config.full_tech or shop.refreshes >= 4:
                raise ValueError('refresh unavailable')
            state = self.apply(state,ResourceDelta(gold=-4*(shop.refreshes+1)),'shop','refresh')
            shop = replace(shop,refreshes=shop.refreshes+1)
            state, shop = self.stock_shop(state,node,shop)
        elif op == 'cultivate':
            state = self.remove(state,option.instance_id,'consume','cultivate')
            rng, state = self.roll(state,'cultivate')
            if rng.random() < self.config.seed_part_probability:
                state = self.grant(state,'pool:PART',source='cultivate')
            else:
                state = self.entry(state,'cross_run_reward',source='cultivate:offspring')
        elif op == 'leave':
            return self.finish(state,node)
        state = self.save_shop(state,shop)
        return self.set_options(state,node,self.shop_options(state,node))

    def movement_targets(self, state, walking):
        if state.resources.parts > state.parts_capacity:
            return {}
        if not state.equipped_instance_id:
            return walking
        item = next(x for x in state.item_instances if x.instance_id == state.equipped_instance_id)
        definition = self.catalog.items[item.item_id]
        origin = state.floor_map.node(state.current_node_id)
        result = {}
        cost = int(definition.move_ap or 0)
        if item.uses_remaining == 1 and 'rogue_6_relic_cargo_6' in state.inventory:
            cost = 0
        if definition.random_move:
            # Public action is an activation at the current node. Choosing an
            # eligible destination ID here leaks unrevealed node types.
            # Ferocity/mystery is public even before exact node revelation.
            # An empty pool cannot execute a movement; don't offer that action.
            return ({state.current_node_id:cost}
                    if self.random_transport_candidates(state) else {})
        for node in state.floor_map.nodes:
            if node.node_id == origin.node_id and node.node_type not in SHOPS:
                continue
            if definition.move_range and (node.row-origin.row,node.col-origin.col) not in definition.move_range:
                continue
            targets = definition.move_target_types
            if 'ALL' not in targets:
                if 'EVENTS' in targets:
                    if node.is_battle:
                        continue
                elif node.node_type.value not in targets:
                    continue
                if item.item_id == PREFIX+'M_10' and node.node_id not in state.revealed:
                    continue
            # For random transport, expose one action; never let policy select RNG destination.
            result[node.node_id] = cost
        return result

    def random_transport_candidates(self, state):
        # Use only the public coarse category for eligibility. Exact hidden
        # event identities must never influence the activation mask.
        # Xianshu #9, BV1sr4y6TEnY 00:27-00:35: M_07 can return to the
        # current FINAL. The item says any non-combat node, with no exclusion
        # for the origin. Unknown-first priority is applied after this pool.
        return tuple(node for node in state.floor_map.nodes
                     if node.hidden_category == 'mystery')

    def on_move(self, state, node, equipment):
        state = relic_effects.on_move_appraisal(self,state)
        state = replace(state,movement_count=state.movement_count+1)
        if equipment:
            instance = next(x for x in state.item_instances if x.instance_id == equipment)
            if instance.item_id == PREFIX+'M_09':
                state = self.apply(state,ResourceDelta(hope=2),'move:old_mother')
            elif instance.item_id == PREFIX+'M_10':
                state = self.apply(state,ResourceDelta(gold=4),'move:cannots_tentacle')
            elif instance.item_id == PREFIX+'M_12':
                state = self.apply(state,ResourceDelta(action_points=3),'move:remote')
            if instance.uses_remaining == 1:
                state = self.remove(state,equipment,'consume','movement')
                state = relic_effects.on_vehicle_exhausted(self,state)
            else:
                state = replace(state,item_instances=tuple(replace(x,uses_remaining=x.uses_remaining-1)
                    if x.instance_id == equipment else x for x in state.item_instances))
            for item in tuple(state.item_instances):
                trigger = False
                if item.item_id == PREFIX+'P_01' and node.is_battle:
                    # White bird pays after the fight, not before its resolution.
                    key=node.node_id+':white_bird_rewards'
                    state=self.set_counter(state,key,self.counter(state,key)+1)
                    state=self.remove(state,item.instance_id,'consume','passive')
                elif item.item_id == PREFIX+'P_02' and not node.is_battle:
                    trigger = True
                elif item.item_id == PREFIX+'P_05' and node.node_type == NodeType.WISH:
                    trigger = True
                elif item.item_id == PREFIX+'P_06' and node.node_type == NodeType.SACRIFICE:
                    trigger = True
                elif item.item_id == PREFIX+'P_03':
                    state = self.apply(state,ResourceDelta(shield=3),'passive:white_fish')
                    state = self.remove(state,item.instance_id,'consume','passive')
                if trigger:
                    state = self.grant(state,'pool:RELIC',2 if item.item_id == PREFIX+'P_06' else 1,source='passive:'+item.item_id)
                    if item.item_id in (PREFIX+'P_01',PREFIX+'P_02'):
                        state = self.remove(state,item.instance_id,'consume','passive')
        return state

    def on_reveal(self, state, count):
        if count:
            state = replace(state,item_instances=tuple(replace(x,appraisal=x.appraisal+count)
                if x.item_id == PREFIX+'G_03' else x for x in state.item_instances))
        return state

    def on_floor_exit(self, state):
        for item in tuple(state.item_instances):
            if self.catalog.items[item.item_id].expires_on_floor_change:
                state = self.remove(state,item.instance_id,'consume','floor_expiry')
        if self.config.difficulty >= 9:
            state = self.apply(state,ResourceDelta(gold=-math.floor(state.resources.gold*0.1)),'difficulty9:floor_gold_loss')
        return state

    def item_choices(self,state,category,count,source,*,pool_id=None):
        from .economic_sampling import choose_from_pool,source_pool,source_pool_assumption
        if pool_id is None:
            assumption = source_pool_assumption(category, source, source_pool(category, source))
            if assumption:
                state = self.entry(state, 'model_assumption', source=assumption)
        pool_id=pool_id or source_pool(category,source)
        rng,state=self.roll(state,'choices:'+source)
        selected=[]
        for _ in range(count):
            candidates=self.candidates(state,category,exclude=selected)
            # Even duplicate-compatible scraps should be distinct alternatives.
            candidates=[x for x in candidates if x not in selected]
            item_id,sampling=choose_from_pool(self.catalog,rng,candidates,
                pool_id=pool_id,difficulty=self.config.difficulty)
            if item_id is None:
                break
            selected.append(item_id)
            state=self.entry(state,'sampling',source=(pool_id or source)+':'+sampling)
        return state,selected

    def reward_menu(self, state, node):
        """Resolve separate reward slots, one chosen item per slot."""
        category = 'RELIC' if self.counter(state,node.node_id+':relic_rewards') else 'PART'
        count = self.counter(state,node.node_id+(':relic_rewards' if category=='RELIC' else ':part_rewards'))
        if not count:
            recruits=self.reward_recruit_options(state)
            return self.set_options(state,node,recruits) if recruits else self.finish(state,node)
        choice_count = (2 if self.config.full_tech else 1) if category == 'PART' else 2
        if category == 'RELIC' and 'rogue_6_relic_legacy_113' in state.inventory:
            choice_count += 1
        source='battle_boss' if node.node_type==NodeType.BATTLE_BOSS else 'battle_elite' if node.node_type in (NodeType.BATTLE_ELITE,NodeType.BATTLE_SAVAGE) else 'battle_normal'
        pool_id=None
        if node.stage_id and category=='RELIC':
            from .events import SPECIAL_BATTLES
            from .residents import RESIDENT_BATTLES
            pool_id=SPECIAL_BATTLES.get(node.stage_id,{}).get('relic_pool')
            if node.stage_id in RESIDENT_BATTLES:
                pool_id=RESIDENT_BATTLES[node.stage_id].relic_pool
        state,candidates=self.item_choices(state,category,choice_count,source,pool_id=pool_id)
        options = tuple(EventOption('take:'+item_id,self.catalog.items[item_id].name,
            operation='take',item_id=item_id,ends_node=False) for item_id in candidates)
        if not options:
            key = node.node_id+(':relic_rewards' if category=='RELIC' else ':part_rewards')
            return self.reward_menu(self.set_counter(state,key,0),node)
        return self.set_options(state,node,options+self.reward_recruit_options(state))

    def battle(self, state, node, *, special_gold=None, chase=False,reward_floor=None):
        normal = (1,2,2,2,2,5)
        elite = (2,2,3,3,5,5)
        boss = (0,0,5,0,8,8)
        reward_floor=reward_floor or state.floor
        from .battle_loot import choose_stage,stage_candidates,draw_chests,resolve_chests
        if node.stage_id is None and stage_candidates(reward_floor,node.node_type):
            rng,state=self.roll(state,'battle_stage')
            stage=choose_stage(rng,floor=reward_floor,node_type=node.node_type)
            node=replace(node,stage_id=stage.stage_id)
            state=self.entry(state,'model_assumption',source='stage:'+stage.selection_evidence)
        tree_parts=0
        soil_parts=0
        from .utopia_economy import extra_battle_scraps
        region=state.region_state
        if region and region.affects(node.node_id):
            soil_parts=extra_battle_scraps(region.ideology,won=True)
        if node.stage_id:
            rng,state=self.roll(state,'battle_chests')
            draws=draw_chests(rng,node.stage_id)
            for draw in draws:
                state=self.entry(state,'chest_spawn',quantity=int(draw.outcome!='none'),
                    source=f'{node.stage_id}:{draw.group_id}:{draw.outcome}:{draw.evidence}')
            loot=resolve_chests(draws,assume_all_opened=self.config.assume_all_chests_opened,
                collected_groups=None if self.config.assume_all_chests_opened else ())
            if not self.config.assume_all_chests_opened:
                # No observed collection was supplied: this is a conservative
                # scenario setting, not a replay observation of empty rewards.
                loot=replace(loot,outcome_evidence='ASSUME_NO_CHESTS_COLLECTED')
            if draws:
                state=self.entry(state,'model_assumption',source='chest_collection:'+loot.outcome_evidence)
            state=self.apply(state,ResourceDelta(gold=loot.gold),'battle_chests')
            tree_parts=loot.scraps
        gold = special_gold if special_gold is not None else (
            boss[reward_floor-1] if node.node_type == NodeType.BATTLE_BOSS else
            elite[reward_floor-1] if node.node_type in (NodeType.BATTLE_ELITE,NodeType.BATTLE_SAVAGE) else normal[reward_floor-1])
        state=self.battle_gold(state,gold,'battle')
        state=self.apply(state,ResourceDelta(tickets=1),'battle')
        from .events import SPECIAL_BATTLES
        from .residents import RESIDENT_BATTLES
        if (node.node_type in (NodeType.BATTLE_NORMAL,NodeType.BATTLE_ELITE,NodeType.BATTLE_BOSS)
                and not chase and not node.event_name and special_gold is None
                and node.stage_id not in SPECIAL_BATTLES and node.stage_id not in RESIDENT_BATTLES):
            from .recruitment_candidates import make_battle_recruit_group
            rng,state=self.roll(state,'ordinary_battle_recruit_ticket')
            group=make_battle_recruit_group(rng,group_id=f'r{state.recruit_candidate_serial}',
                node_type=node.node_type,radio_owned='rogue_6_relic_legacy_21' in state.inventory,
                observed_base_count=dict(self.config.observed_battle_recruit_candidate_counts).get(node.node_type.value))
            state=replace(state,pending_recruit_candidate_groups=state.pending_recruit_candidate_groups+(group,),
                recruit_candidate_serial=state.recruit_candidate_serial+1)
            state=self.entry(state,'recruit_candidate_group_granted',instance_id=group.group_id,
                quantity=1,source=group.source+':'+group.base_count_evidence)
            state=self.entry(state,'recruit_candidate_offered',instance_id=group.group_id,
                quantity=len(group.candidate_item_ids),source=f'{group.pool_id}:{group.composition_evidence}:'
                f'{group.sampling_assumption}:n={group.sample_count}'+
                (':CLIENT:battle_extra_recruit_ticket:count=1' if group.radio_owned else ''))
        elif node.stage_id in RESIDENT_BATTLES and not chase:
            from .recruitment import add_unknown_reserve_opportunities
            state=add_unknown_reserve_opportunities(self,state,'resident',1)
        else:
            state=self.entry(state,'unresolved_recruit_ticket_type',source='special_battle:generic_acquisition_only')
        state = replace(state,battle_count=state.battle_count+1)
        state = relic_effects.on_battle(self,state,perfect=None,chase=chase,stage_id=node.stage_id)
        from .progression import award_battle_exp,battle_base_exp
        state=award_battle_exp(self,state,battle_base_exp(node,reward_floor))
        white_birds=self.counter(state,node.node_id+':white_bird_rewards')
        if white_birds:
            state=self.grant(state,'pool:RELIC',white_birds,source='passive:'+PREFIX+'P_01')
            state=self.set_counter(state,node.node_id+':white_bird_rewards',0)
        if tree_parts:
            state=self.grant(state,'pool:PART',tree_parts,source='tree_chest')
        if soil_parts:
            state=self.grant(state,'pool:PART',soil_parts,source='hope_soil')
        rng, state = self.roll(state,'battle_drop_presence')
        relic = node.node_type in (NodeType.BATTLE_ELITE,NodeType.BATTLE_SAVAGE,NodeType.BATTLE_BOSS) or rng.random() < self.config.normal_relic_probability
        from .residents import RESIDENT_BATTLES
        if node.stage_id in RESIDENT_BATTLES:
            spec=RESIDENT_BATTLES[node.stage_id]
            probability=self.config.stronghold_relic_probability if spec.stronghold else self.config.normal_relic_probability
            relic=rng.random()<probability
            state=self.entry(state,'model_assumption',source='residents:unverified_relic_presence_probability')
        part = rng.random() < self.config.battle_part_probability
        state = self.set_counter(state,node.node_id+':relic_rewards',int(relic))
        state = self.set_counter(state,node.node_id+':part_rewards',int(part))
        if node.node_type == NodeType.BATTLE_BOSS:
            state = replace(state,awaiting_exit=True)
        return self.reward_menu(state,node)

    def wish_menu(self, state, node):
        count = 2 + int('rogue_6_relic_legacy_113' in state.inventory)
        refreshed=self.counter(state,node.node_id+':refresh')>0
        pool_id='rogue_6:得偿所愿（刷新）' if refreshed and 'rogue_6:得偿所愿（刷新）' in self.catalog.observed_pools else 'rogue_6:得偿所愿'
        state,selected=self.item_choices(state,'RELIC',count,'wish',pool_id=pool_id)
        options = [EventOption('wish:'+item_id,self.catalog.items[item_id].name,
            operation='event_reward',item_id=item_id) for item_id in selected]
        if self.counter(state,node.node_id+':refresh') == 0:
            options.append(EventOption('wish_refresh','4锭刷新一次',ResourceDelta(gold=-4),
                operation='wish_refresh',price=4,ends_node=False))
        options.append(EventOption('leave','离开',operation='leave'))
        return self.set_options(state,node,options)

    def sacrifice_menu(self,state,node):
        options=[]
        count=self.counter(state,node.node_id+':exchanges')
        chosen_kind=self.counter(state,node.node_id+':exchange_kind')
        if count < (2 if self.config.full_tech else 1):
            for item in state.item_instances:
                definition=self.catalog.items[item.item_id]
                if not definition.can_sacrifice:
                    continue
                kind=1 if item.category=='RELIC' else 2
                if chosen_kind and chosen_kind!=kind:
                    continue
                if kind==2 and not self.config.full_tech:
                    continue
                options.append(EventOption('exchange:'+item.instance_id,
                    '置换'+definition.name,operation='exchange',item_id=item.item_id,
                    instance_id=item.instance_id,ends_node=False))
        options.append(EventOption('leave','离开',operation='leave'))
        return self.set_options(state,node,options)

    def enter(self,state,node):
        state=relic_effects.on_node_enter(self,state,node)
        context=state.resident_context
        if context and (node.node_id in context.occupied_node_ids or
                node.node_id in context.stronghold_node_ids-context.defeated_stronghold_ids):
            from .residents import select_resident_battle
            rng,state=self.roll(state,'resident_encounter_stage')
            spec=select_resident_battle(stronghold=node.node_id in context.stronghold_node_ids,
                rng=rng,allow_synthetic=True)
            if node.node_type==NodeType.LIGHT and node.node_id not in state.completed:
                state=self.apply(state,ResourceDelta(action_points=1),'occupied_light')
            state=self.set_counter(state,node.node_id+':resident_battle',1)
            state=self.entry(state,'model_assumption',source='residents:unverified_uniform_stage')
            return self.battle(state,replace(node,node_type=NodeType.BATTLE_NORMAL,event_name=None,
                stage_id=spec.stage_id),special_gold=spec.gold)
        if node.node_type in SHOPS:
            return self.enter_shop(state,node)
        if node.node_type in (NodeType.FINAL,NodeType.EVACUATE):
            return self.set_options(state,node,(EventOption('advance','领取加工品并前往下一区域',operation='advance'),
                                               EventOption('leave','暂不离开本区域',operation='leave'))+
                (self.recruitment_options(state,ticket_instance_ids=state.stored_recruit_ticket_ids)
                 if node.node_type==NodeType.FINAL else ()))
        if node.node_type==NodeType.STORY and state.floor==5 and state.fate_context is not None:
            from .fate_nodes import enter_fate,TRUE_FATE_NAME,sample_fate_belief
            from .events import documented_fate_options,documented_true_fate_options
            context=state.fate_context
            if context.true_node_id is None:
                rng,state=self.roll(state,'fate_belief')
                context=sample_fate_belief(context,rng=rng,allow_synthetic=True)
            context,scene=enter_fate(context,node.node_id)
            state=replace(state,fate_context=context)
            node=replace(node,event_name=scene)
            if scene==TRUE_FATE_NAME:
                options=documented_true_fate_options(already_entered=node.node_id in state.completed)
                return self.set_options(state,replace(node,repeatable=True),options)
            options=documented_fate_options(scene,state,real_fate_marked=context.true_fate_marked)
            return self.set_options(state,node,options) if options else self.finish(state,node)
        if node.node_id in state.completed:
            return replace(state,pending_node_id=None)
        if node.is_battle:
            return self.battle(state,node)
        if node.node_type == NodeType.WISH:
            return self.wish_menu(state,node)
        if node.node_type == NodeType.SACRIFICE:
            return self.sacrifice_menu(state,node)
        if node.node_type == NodeType.INCIDENT:
            from .events import documented_incident_options,eligible_incident_names
            eligible=eligible_incident_names(state.floor,state,self.event_pools[state.floor])
            named_is_eligible=(node.event_name and node.event_name!='未进入节点期望模型' and
                eligible_incident_names(state.floor,state,(node.event_name,)))
            if not named_is_eligible:
                if not eligible:
                    return self.finish(state,node)
                rng,state=self.roll(state,'unobserved_incident')
                node=replace(node,event_name=rng.choice(eligible))
            return self.set_options(state,node,documented_incident_options(node.event_name or '',state))
        if node.node_type == NodeType.REST:
            if self.config.full_tech:
                state=self.apply(state,ResourceDelta(hp=3),'full_tech:rest_entry')
            options=[EventOption('rest_hp','生命上限+3',ResourceDelta(max_hp=3),operation='event_reward'),
                     EventOption('rest_hope','希望+3',ResourceDelta(hope=3),operation='event_reward'),
                     EventOption('rest_ap','行动力+2',ResourceDelta(action_points=2),operation='event_reward'),
                     EventOption('rest_squad','编队容量+1',operation='squad_capacity'),
                     EventOption('rest_parts','零件箱容量+1',operation='parts_capacity'),
                     EventOption('rest_voucher','高级物资配给券',operation='event_reward',
                                 item_id='rogue_6_upgrade_ticket_all')]
            rng,state=self.roll(state,'rest_menu')
            return self.set_options(state,node,rng.sample(options,3))
        if node.node_type == NodeType.LIGHT:
            return self.finish(self.apply(state,ResourceDelta(action_points=1),'light'),node)
        if node.node_type in (NodeType.EMPTY,NodeType.START,NodeType.DOOR):
            return self.finish(state,node)
        if node.node_type == NodeType.DUEL:
            if state.portal_context and state.portal_context.variation_id==8:
                # A victory oracle cannot certify fresh formal operator teams.
                return self.set_options(state,node,(
                    EventOption('roster','指定尚未力竭的参战干员',operation='needs_observation'),
                    EventOption('portal_return','结束黑潭探索，返回入口',operation='portal_return')))
            return self.set_options(state,node,(
                EventOption('duel_left','左侧奖励：零件及2招募券',ResourceDelta(tickets=2),operation='event_reward',item_id='pool:rogue_6:狭路相逢：左'),
                EventOption('duel_middle','中间奖励：零件及2招募券',ResourceDelta(tickets=2),operation='event_reward',item_id='pool:rogue_6:狭路相逢：中'),
                EventOption('duel_right','右侧奖励：藏品及2招募券',ResourceDelta(tickets=2),operation='event_reward',item_id='pool:rogue_6:狭路相逢：右')))
        if node.node_type == NodeType.EXPEDITION:
            return self.expedition_menu(state,node)
        if node.node_type == NodeType.EMPLOY:
            from .observed_operator_bridge import employment_menu
            return employment_menu(self,replace(state,pending_node_id=node.node_id))
        if node.node_type == NodeType.PORTAL:
            from .portal import portal_entry_options
            rng,state=self.roll(state,'portal_entrance_offer')
            offered=portal_entry_options(state,rng)
            options=[EventOption('portal:'+instance_id,'消耗加工品，进入黑潭',
                operation='portal_enter',instance_id=instance_id,
                item_id=next(x.item_id for x in state.item_instances if x.instance_id==instance_id))
                for instance_id in offered]
            options.append(EventOption('leave','离开',operation='leave'))
            return self.set_options(state,node,options)
        return self.set_options(state,node,(
            EventOption('observe','继续：需真实场景',operation='needs_observation',ends_node=False),
            EventOption('leave','离开',operation='leave')))

    def choose(self,state,node,option):
        op=option.operation
        context=state.chase_reward_context
        if context is not None:
            if (node.node_id!=context.carrier_node.node_id or state.pending_node_id!=node.node_id
                    or option not in state.floor_map.node(node.node_id).options):
                raise ValueError('chase choice requires its current fixed reward menu')
            if op=='take':
                if option in context.relic_options:
                    context=replace(context,relic_options=())
                elif option in context.part_options:
                    context=replace(context,part_options=())
                else:
                    raise ValueError('chase reward was already claimed')
                state=replace(state,chase_reward_context=context)
                previous_instances={item.instance_id for item in state.item_instances}
                state=self.acquire(state,option.item_id,'boss_chase')
                # A taken collectible can itself issue concrete recruitment
                # or upgrade tickets. Those belong to this reward phase, too;
                # never infer opportunities from historical abstract counts.
                granted=tuple(item.instance_id for item in state.item_instances
                    if item.instance_id not in previous_instances
                    and item.category in ('RECRUIT_TICKET','UPGRADE_TICKET'))
                state=replace(state,pending_recruit_ticket_ids=tuple(dict.fromkeys(
                    state.pending_recruit_ticket_ids+granted)))
                return self.chase_reward_menu(state)
            if op not in ('select_recruit_ticket','recruit_reserve','recruit_temporary',
                          'mechanist_promote','retain_recruit_ticket','decline_recruitment'):
                raise ValueError('map interaction is suspended during chase rewards')
        if op in ('broker_reserve','broker_random_six'):
            if (node.event_name != '临时中介所' or node.node_type != NodeType.INCIDENT
                    or state.pending_node_id != node.node_id or node.node_id in state.completed
                    or option not in state.floor_map.node(node.node_id).options
                    or self.counter(state,node.node_id+':broker_branch')):
                raise ValueError('broker branch requires its live unclaimed incident menu')
            state=self.set_counter(state,node.node_id+':broker_branch',1 if op=='broker_reserve' else 2)
            if op=='broker_random_six':
                state=self.entry(state,'coverage_required',source=
                    'choice_ro6_hire1_2:random_six_star_identity_and_duplicate_resolution_unobserved')
                return self.set_options(state,node,(),observation=True)
            from .recruitment import add_unknown_reserve_opportunities,BROKER_RESERVE_SOURCE
            state=self.entry(state,'model_abstraction',source=
                'broker_guaranteed_reserve:common_action_of_four_client_variants:profession_and_other_choices_unmodelled')
            state=self.apply(state,ResourceDelta(hope=2,tickets=1),'broker_guaranteed_reserve:branch_selected')
            state=add_unknown_reserve_opportunities(self,state,BROKER_RESERVE_SOURCE,1)
            return self.after_recruitment_choice(state,node)
        if op=='recruit_temporary':
            from .temporary_recruitment import resolve_temporary_recruit,load_pinned_temporary_candidate_catalog
            from .observed_operator_bridge import emergency_character_ids
            offer=next((x for x in state.temporary_recruit_offers
                        if x.ticket_instance_id==option.instance_id),None)
            if offer is None or option not in self.recruitment_options(state):
                raise ValueError('temporary recruitment requires its actual observed identity')
            state=resolve_temporary_recruit(self,state,load_pinned_temporary_candidate_catalog(),offer,option,
                blocked_operator_ids=emergency_character_ids(state))
            return self.after_recruitment_choice(state,node)
        if op in ('emergency_hire','employment_refresh'):
            from .observed_operator_bridge import employment_menu,formal_character_ids
            from .employment import hire_emergency,next_employment_refresh_cost
            if (node.node_type!=NodeType.EMPLOY or state.employment_context is None
                    or self.counter(state,node.node_id+':employment_refresh_paid')
                    or option not in employment_menu(self,state).floor_map.node(node.node_id).options):
                raise ValueError('employment transaction requires its live observed menu')
            if op=='emergency_hire':
                change=hire_emergency(state.employment_context,option.option_id,
                    gold=state.resources.gold,present_char_ids=formal_character_ids(state))
                state=self.apply(state,ResourceDelta(gold=-change.gold_spent),'observed_emergency_hire','purchase')
                state=replace(state,employment_context=change.context)
                state=self.set_counter(state,'emergency_hired_after_battle:'+change.hired.instance_id,state.battle_count)
                state=self.entry(state,'emergency_hire',quantity=1,instance_id=change.hired.instance_id,
                    source='OBSERVED_MENU:'+change.hired.char_id)
                # User-confirmed 2026-09-09: each successful emergency hire grows held corn.
                state=relic_effects.on_recruit(self,state,recruited_count=1)
            else:
                cost=next_employment_refresh_cost(state.employment_context)
                if cost is None or option.price!=cost:
                    raise ValueError('employment refresh is exhausted or stale')
                state=self.apply(state,ResourceDelta(gold=-cost),'employment_refresh','refresh')
                state=self.set_counter(state,node.node_id+':employment_refresh_paid',cost)
            return employment_menu(self,state)
        if op=='select_recruit_ticket':
            from .recruitment_candidates import recruit_candidate_options,resolve_recruit_candidate_option
            group=next((group for group in state.pending_recruit_candidate_groups
                if option in recruit_candidate_options(group,name_of=lambda item_id:self.catalog[item_id].name)),None)
            if group is None or state.pending_node_id is None:
                raise ValueError('selecting a ticket requires a live unresolved candidate group')
            _,item_id=resolve_recruit_candidate_option(group,option,name_of=lambda item_id:self.catalog[item_id].name)
            state=replace(state,pending_recruit_candidate_groups=tuple(
                x for x in state.pending_recruit_candidate_groups if x.group_id!=group.group_id))
            instance_id=f'i{state.item_serial}'
            state=self.acquire(state,item_id,'battle_recruit_ticket:'+group.pool_id)
            state=replace(state,pending_recruit_ticket_ids=state.pending_recruit_ticket_ids+(instance_id,))
            state=self.entry(state,'recruit_candidate_selected',instance_id=group.group_id,quantity=1,
                source=f'{group.source}:selected_ticket:{instance_id}:{item_id}')
            return self.after_recruitment_choice(state,node)
        if op=='recruit_reserve':
            from .recruitment import resolve_reserve_recruit,resolve_unknown_reserve_recruit
            if option not in self.recruitment_options(state):
                raise ValueError('stored recruitment tickets are only usable at FINAL')
            resolve=(resolve_unknown_reserve_recruit if option.instance_id in
                {x.opportunity_id for x in state.unknown_recruit_opportunities} else resolve_reserve_recruit)
            return self.after_recruitment_choice(resolve(self,state,option),node)
        if op in ('retain_recruit_ticket','decline_recruitment'):
            if option not in self.reward_recruit_options(state):
                raise ValueError('recruitment disposition requires a real pending ticket and available capacity')
            if any(x.ticket_instance_id==option.instance_id for x in state.temporary_recruit_offers):
                from .temporary_recruitment import invalidate_temporary_offer
                state=invalidate_temporary_offer(self,state,option.instance_id,reason=op)
                state=replace(state,temporary_recruit_offers=tuple(x for x in state.temporary_recruit_offers
                                                               if x.ticket_instance_id!=option.instance_id))
            if op=='retain_recruit_ticket':
                state=replace(state,pending_recruit_ticket_ids=tuple(x for x in state.pending_recruit_ticket_ids
                    if x!=option.instance_id),stored_recruit_ticket_ids=state.stored_recruit_ticket_ids+(option.instance_id,))
                state=self.entry(state,'recruit_ticket_retained',instance_id=option.instance_id,quantity=1,
                    source=('USER_OBSERVED_RULE:free_reserve_option:' if option.instance_id.startswith('u') else '')+option.instance_id)
            else:
                if option.instance_id in {x.opportunity_id for x in state.unknown_recruit_opportunities}:
                    from .recruitment import discard_unknown_reserve_opportunity
                    state=discard_unknown_reserve_opportunity(self,state,option.instance_id)
                else:
                    state=self.remove(state,option.instance_id,'discard','declined_recruitment')
            return self.after_recruitment_choice(state,node)
        if op=='mechanist_promote':
            from .operator_economy import MECHANIST_ID,mechanist_eligible_ticket_ids,mechanist_hope_cost,promote_formal
            ticket=next((item for item in state.item_instances if item.instance_id==option.instance_id),None)
            if ticket is not None and ticket.category=='UPGRADE_TICKET':
                from .upgrade_ticket_actions import resolve_mechanist_upgrade_ticket
                return self.after_recruitment_choice(resolve_mechanist_upgrade_ticket(self,state,option),node)
            if ticket is None or ticket.item_id not in mechanist_eligible_ticket_ids():
                raise ValueError('promotion requires an actual eligible held recruitment ticket')
            if option not in self.recruitment_options(state):
                raise ValueError('stored recruitment tickets are only usable at FINAL')
            if MECHANIST_ID in state.promoted_operator_ids:
                raise ValueError('mechanist already promoted; do not consume another ticket')
            cost=mechanist_hope_cost(promotion=True,full_mastery=self.config.mechanist_full_mastery)
            state=promote_formal(self,state,MECHANIST_ID,source='recruit_ticket:'+ticket.item_id,hope_cost=cost)
            return self.after_recruitment_choice(self.remove(state,ticket.instance_id,'consume','mechanist_promotion'),node)
        if op=='starting_reward':
            from .ending_rules import STARTING_REWARD_BY_ID
            if self.counter(state,'starting_reward_selected'):
                raise ValueError('starting reward already selected')
            spec=STARTING_REWARD_BY_ID[option.option_id]
            state=self.apply(state,ResourceDelta(gold=spec.gold,max_hp=spec.max_hp),'starting_reward:'+spec.choice_id)
            state=replace(state,parts_capacity=state.parts_capacity+spec.parts_capacity)
            if spec.item_category:
                item_pool='pool:'+spec.item_category
                if spec.client_rarity_label:
                    item_pool+=':'+{'普通':'NORMAL','稀有':'RARE'}[spec.client_rarity_label]
                state=self.entry(state,'model_assumption',source='starting_reward:unverified_item_pool:'+spec.choice_id)
                state=self.grant(state,item_pool,spec.item_count,source='starting_reward:'+spec.choice_id)
            state=self.set_counter(state,'starting_reward_selected',1)
            state=self.set_counter(state,'starting_reward_pending',0)
            state=self.initialize_formal_roster(state)
            return self.finish(state,node)
        if op in ('fate_mark','fate_leave'):
            from .fate_nodes import mark_true_fate
            if state.fate_context is None:
                raise ValueError('fate marker requires a known generated fate group')
            state=self.apply(state,option.effect,'fate_mark')
            state=replace(state,fate_context=mark_true_fate(state.fate_context))
            state=self.entry(state,'fate_marker',source=str(option.item_id))
            return self.finish(state,replace(node,repeatable=op=='fate_leave'))
        if op=='fate_commit':
            from .events import documented_true_fate_options
            boss_node=replace(node,node_type=NodeType.BATTLE_BOSS,stage_id='ro6_b_5',repeatable=False)
            return self.set_options(state,boss_node,documented_true_fate_options(committed_to_battle=True))
        if op=='fate_battle':
            if node.node_type!=NodeType.BATTLE_BOSS or node.stage_id!='ro6_b_5':
                raise ValueError('true fate battle requires its committed boss node')
            return self.battle(state,node)
        if op=='cave_draw':
            from .events import cave_draw_outcomes,documented_incident_options
            rng,state=self.roll(state,'cave_draw')
            state=self.entry(state,'model_assumption',source='cave:unverified_equal_outcome_probability')
            outcome=rng.choice(cave_draw_outcomes(option.quantity,portal=state.portal_context is not None))
            if outcome.battle:
                return self.battle(state,replace(node,node_type=NodeType.BATTLE_NORMAL),
                    reward_floor=min(6,state.floor+1))
            state=self.apply(state,outcome.effect,'event:洞中宝')
            if outcome.item_id:
                state=self.grant(state,outcome.item_id,outcome.quantity,source='event:洞中宝')
            if outcome.ends_node:
                return self.finish(state,node)
            state=self.set_counter(state,node.node_id+':stage',option.quantity)
            return self.set_options(state,node,documented_incident_options(node.event_name,state))
        if op=='red_moss':
            if option.instance_id:
                state=self.remove(state,option.instance_id,'consume','red_moss:seed')
            state=self.apply(state,option.effect,'event:呼吸的红苔')
            state=self.set_counter(state,'next_hope_soil',1)
            return self.finish(state,node)
        if op in ('remembrance_parts','remembrance_key'):
            from .rewind import pay_remembrance_item_cost
            state=pay_remembrance_item_cost(self,state,op)
            return self.finish(state,node)
        if op=='portal_enter':
            from .portal import build_portal_layout,load_portal_evidence,enter_portal_state
            state=self.remove(state,option.instance_id,'consume','portal_entry')
            state=self.finish(state,node)
            rng,state=self.roll(state,'portal_zone')
            count=self.counter(state,'portal_entries')+1
            state=self.set_counter(state,'portal_entries',count)
            zone=rng.choice(tuple(load_portal_evidence()['zone_action_points']))
            layout=build_portal_layout(zone,outer_floor=state.floor,seed=rng.getrandbits(63),
                namespace=f'P{count}',allow_synthetic=True)
            state=self.entry(state,'model_assumption',source='portal:unverified_uniform_zone_probability')
            for assumption in layout.assumptions:
                state=self.entry(state,'model_assumption',source='portal:'+assumption)
            state,context=enter_portal_state(state,layout)
            # Entering a different map reveals its start before any new rewards.
            state=self.on_reveal(state,len(state.revealed))
            return self.initialize_population(replace(state,portal_context=context))
        if op=='portal_return':
            state=replace(state,pending_node_id=None)
            return self.return_portal(state)
        if op=='special_battle':
            from .events import SPECIAL_BATTLES
            stage=option.item_id
            rng,state=self.roll(state,'special_battle')
            if stage=='stage_pool:black_birth':
                stage=rng.choice(('ro6_t_13','ro6_t_14','ro6_t_15'))
            spec=SPECIAL_BATTLES[stage]
            base_gold=rng.randint(spec['gold'],spec['gold_max'])
            state=self.entry(state,'sampling',source='special_battle:unverified_extra_drop_prior')
            battle_node=replace(node,node_type=NodeType.BATTLE_ELITE if spec['is_elite'] else NodeType.BATTLE_NORMAL,stage_id=stage)
            return self.battle(state,battle_node,special_gold=base_gold)
        if op in ('event_advance','lake_resolve','shadow_dance'):
            from .events import documented_incident_options,shadow_dance_outcomes
            state=self.apply(state,option.effect,'event:'+str(node.event_name))
            if option.item_id:
                state=self.grant(state,option.item_id,option.quantity,source='event:'+str(node.event_name))
            stage=self.counter(state,node.node_id+':stage')
            if op=='event_advance':
                state=self.set_counter(state,node.node_id+':stage',stage+1)
            elif op=='lake_resolve':
                rng,state=self.roll(state,'lake_resolution')
                state=self.entry(state,'model_assumption',source='lake:unverified_equal_branch_probability')
                if rng.random()<0.5:
                    return self.choose(state,node,EventOption('lake_battle','紧急湖中魇',operation='special_battle',item_id='ro6_e_t_5'))
                return self.finish(self.grant(state,'pool:RELIC:SUPER_RARE',source='lake_offering'),node)
            else:
                outcomes=shadow_dance_outcomes(option.quantity,had_relic_on_entry='rogue_6_relic_artifact_4' in state.inventory)
                rng,state=self.roll(state,'shadow_dance')
                state=self.entry(state,'model_assumption',source='shadow_dance:unverified_equal_branch_probability')
                outcome=rng.choice(outcomes)
                state=self.apply(state,outcome.effect,'shadow_dance')
                if outcome.operation=='event_squad_capacity':
                    state=replace(state,squad_capacity=state.squad_capacity+1)
                if outcome.item_id:
                    state=self.grant(state,outcome.item_id,outcome.quantity,source='shadow_dance')
                if option.quantity==8 and outcome.ends_node:
                    return self.finish(state,node)
                state=self.set_counter(state,node.node_id+':stage',min(7,stage+1))
            return self.set_options(state,node,documented_incident_options(node.event_name,state))
        if op in ('event_mark','event_move'):
            target_type=NodeType(option.item_id.split(':',1)[1])
            targets=[n for n in state.floor_map.nodes if n.node_type==target_type and n.node_id not in state.completed]
            if targets:
                target=min(targets,key=lambda n:abs(n.row-node.row)+abs(n.col-node.col))
                state=self.on_reveal(state,int(target.node_id not in state.revealed))
                state=replace(state,revealed=state.revealed|{target.node_id})
                if op=='event_move':
                    state=self.finish(state,node)
                    state=replace(state,current_node_id=target.node_id)
                    return self.enter(state,target)
            return self.finish(state,node)
        if op in ('purchase','sell','refresh','cultivate','bank_withdraw') or (node.node_type in SHOPS and op=='leave'):
            return self.transact(state,node,option)
        if op=='needs_observation':
            return self.set_options(state,node,(),observation=True)
        if op=='event_reward':
            if node.node_type==NodeType.DUEL:
                if (self.counter(state,node.node_id+':duel_resolved') or
                        option not in state.floor_map.node(node.node_id).options):
                    raise ValueError('duel settlement requires its live unclaimed branch')
                state=self.set_counter(state,node.node_id+':duel_resolved',1)
                from .progression import award_battle_exp
                rng,state=self.roll(state,'duel_stage')
                state=self.entry(state,'model_assumption',source='duel:unverified_equal_stage_probability')
                state=self.battle_gold(state,3,'duel_battle')
                state=award_battle_exp(self,state,rng.choice((13,18)),source='duel_battle')
                state=relic_effects.on_battle(self,state,perfect=None)
                state=replace(state,battle_count=state.battle_count+1)
                state=self.apply(state,replace(option.effect,parts=0,relics=0),'event:狭路相逢')
                from .recruitment import add_unknown_reserve_opportunities
                state=add_unknown_reserve_opportunities(self,state,'duel',option.effect.tickets)
                # All branches grant one physical reward. The middle branch
                # reveals a jointly observed pair and the player claims one.
                category='RELIC' if option.option_id=='duel_right' else 'PART'
                if option.option_id=='duel_middle':
                    from .duel_rewards import make_duel_middle_reward
                    rng,state=self.roll(state,'duel_middle_candidates')
                    group=make_duel_middle_reward(rng,group_id='duel:'+node.node_id)
                    candidates=group.candidate_item_ids
                    state=self.entry(state,'duel_candidates_offered',instance_id=group.group_id,
                        quantity=2,source=group.evidence+':paired_sample_count='+str(group.paired_sample_count))
                else:
                    state,candidates=self.item_choices(state,category,option.quantity,'duel_reward',
                        pool_id=option.item_id.removeprefix('pool:'))
                key=node.node_id+(':relic_rewards' if category=='RELIC' else ':part_rewards')
                state=self.set_counter(state,key,int(bool(candidates)))
                takes=tuple(EventOption('take:duel:'+str(index)+':'+item_id,self.catalog.items[item_id].name,
                    operation='take',item_id=item_id,ends_node=False) for index,item_id in enumerate(candidates))
                return self.set_options(state,node,takes+self.reward_recruit_options(state))
            state=self.apply(state,replace(option.effect,parts=0,relics=0),'event:'+str(node.event_name))
            if option.item_id:
                state=self.grant(state,option.item_id,option.quantity,source='event:'+str(node.event_name))
            for item_id in option.add_items:
                if item_id in self.catalog.items or item_id.startswith('pool:'):
                    state=self.grant(state,item_id,source='event:'+str(node.event_name))
            return self.finish(state,node)
        if op=='take':
            category=self.catalog.items[option.item_id].category
            key=node.node_id+(':relic_rewards' if category=='RELIC' else ':part_rewards')
            if self.counter(state,key)<=0 or option not in state.floor_map.node(node.node_id).options:
                raise ValueError('reward candidate is already claimed or not currently offered')
            state=self.acquire(state,option.item_id,'event:狭路相逢' if node.node_type==NodeType.DUEL else 'battle')
            state=self.set_counter(state,key,max(0,self.counter(state,key)-1))
            return self.reward_menu(state,node)
        if op=='wish_refresh':
            state=self.apply(state,ResourceDelta(gold=-4),'wish','refresh')
            state=self.set_counter(state,node.node_id+':refresh',1)
            return self.wish_menu(state,node)
        if op=='exchange':
            definition=self.catalog.items[option.item_id]
            state=self.remove(state,option.instance_id,'consume','exchange')
            rank={'NORMAL':'N','RARE':'R','SUPER_RARE':'SR'}[definition.rarity]
            kind='藏品' if definition.category=='RELIC' else '零件'
            state=self.grant(state,f'pool:rogue_6:失与得：{rank}{kind}输出',source='exchange')
            state=self.set_counter(state,node.node_id+':exchanges',self.counter(state,node.node_id+':exchanges')+1)
            state=self.set_counter(state,node.node_id+':exchange_kind',1 if definition.category=='RELIC' else 2)
            return self.sacrifice_menu(state,node)
        if op=='parts_capacity':
            return self.finish(replace(state,parts_capacity=state.parts_capacity+1),node)
        if op=='squad_capacity':
            return self.finish(replace(state,squad_capacity=state.squad_capacity+1),node)
        if op=='supply_voucher':
            return self.finish(replace(state,supply_vouchers=state.supply_vouchers+1),node)
        if op in ('expedition_inside','expedition_source'):
            operator_id=option.item_id
            if operator_id not in state.available_formal_operator_ids:
                raise ValueError('expedition operator is not formally recruited and available')
            if state.portal_context is not None or state.floor not in (2,3,4):
                raise ValueError('expedition is only available on ordinary floors II..IV')
            if op=='expedition_source':
                from .ending_rules import plan_source_expedition
                plan_source_expedition(floor=state.floor,operator_id=operator_id,
                    first_ending_unlocked=self.first_ending_available())
            pending=PendingExpedition(operator_id,op.removeprefix('expedition_'),
                state.floor,state.floor_index+1)
            state=replace(state,expeditions=state.expeditions+(pending,))
            state=self.entry(state,'operator_departed',quantity=1,source=op+':'+operator_id)
            return self.finish(state,node)
        if op=='advance':
            pool = (8,9,10,11,12) if node.node_type==NodeType.EVACUATE else (1,2,3,5,6)
            rng,state=self.roll(state,'floor_exit_part')
            state=self.acquire(state,PREFIX+f'M_{rng.choice(pool):02d}','floor_exit')
            if self.config.full_tech and self.config.difficulty>=9:
                state=self.apply(state,ResourceDelta(max_hp=2),'full_tech:viviparity')
            return replace(self.finish(state,node),awaiting_exit=True)
        if op=='leave':
            # An exit remains usable after declining to leave this floor.
            return self.finish(state,node)
        raise ValueError(f'unsupported economic operation: {op}')

    def transition(self,sim: 'BlackflowSimulator',state,action):
        before=state
        messages=[]
        if action.kind==ActionKind.RETURN:
            state=self.return_portal(state)
            messages.append('返回黑潭入口，恢复区域行动力')
        elif action.kind in (ActionKind.EQUIP,ActionKind.DISCARD):
            option=sim.available_options(state)[action.option_index]
            if action.kind==ActionKind.EQUIP:
                state=replace(state,equipped_instance_id=option.instance_id)
            else:
                state=self.remove(state,option.instance_id,'discard','inventory_overflow')
                if state.chase_reward_context is not None:
                    state=self.chase_reward_menu(state)
            messages.append(option.title)
        elif action.kind==ActionKind.MOVE:
            node=state.floor_map.node(action.target_node_id)
            equipment=state.equipped_instance_id
            origin_id=state.current_node_id
            doors=tuple(n.node_id for n in state.floor_map.nodes if n.node_type==NodeType.DOOR)
            door_pair={doors[0]:doors[1],doors[1]:doors[0]} if len(doors)==2 else {}
            pure_door_hop=(not equipment and state.door_ready_node_id==origin_id and
                door_pair.get(origin_id)==node.node_id and action.movement_cost==0)
            if equipment:
                item=next(x for x in state.item_instances if x.instance_id==equipment)
                if self.catalog.items[item.item_id].random_move:
                    eligible=self.random_transport_candidates(state)
                    if not eligible:
                        raise ValueError('random transport has no eligible destination')
                    unknown=[x for x in eligible if x.node_id not in state.revealed]
                    if self.config.random_transport_unknown_first and unknown:
                        eligible=unknown
                    state=self.entry(state,'model_assumption',source=(
                        'USER_OBSERVED_RULE:random_transport_unknown_first:' if self.config.random_transport_unknown_first
                        else 'random_transport:')+'synthetic_uniform_within_eligible_priority_group')
                    rng,state=self.roll(state,'random_transport')
                    node=rng.choice(eligible)
            state=self.apply(state,ResourceDelta(action_points=-action.movement_cost),'movement')
            if state.region_state is not None:
                from .utopia_economy import apply_policy
                trajectory=(node.node_id,) if equipment else action.traversed_node_ids
                affected=sum(state.region_state.affects(x) for x in trajectory)
                state=apply_policy(self,state,state.region_state.policy,affected_nodes=affected)
            # Appraise only grass already held at the moment of discovery.
            state=self.on_reveal(state,int(node.node_id not in state.revealed))
            state=replace(state,current_node_id=node.node_id,revealed=state.revealed|{node.node_id})
            if not pure_door_hop:
                state=self.move_population(state,node.node_id)
            if state.portal_context and state.portal_context.variation_id==5:
                from .portal import blue_refresh_unvisited
                rng,state=self.roll(state,'blue_portal_refresh')
                maps=list(state.maps)
                maps[state.floor_index]=blue_refresh_unvisited(state.floor_map,visited=state.completed,
                    current_node_id=node.node_id,seed=rng.getrandbits(63),allow_synthetic=True)
                state=replace(state,maps=tuple(maps))
            if not pure_door_hop:
                state=self.on_move(state,node,equipment)
            else:
                state=self.entry(state,'model_assumption',source='door:paid_entry_then_one_free_transfer_no_second_move_trigger')
            route=(origin_id,)+action.traversed_node_ids
            arrived_by_hop=(not equipment and len(route)>=2 and
                door_pair.get(route[-2])==route[-1])
            state=replace(state,door_ready_node_id=(node.node_id
                if node.node_type==NodeType.DOOR and not arrived_by_hop else None))
            messages.append(f'移动到{sim.node_label(node)}（{action.movement_cost}行动力）')
            state=self.enter(state,node)
        else:
            node=state.floor_map.node(state.pending_node_id or state.current_node_id)
            option=sim.available_options(state)[action.option_index]
            state=self.choose(state,node,option)
            messages.append(option.title)
        clear_reward=0.0
        if state.pending_node_id is None and state.resources.parts<=state.parts_capacity:
            if state.chase_reward_pending:
                state,clear_reward,msg=self.complete_chase(sim,state)
                messages.extend(msg)
            elif state.awaiting_exit:
                exit_node=state.floor_map.node(state.current_node_id)
                state=replace(state,awaiting_exit=False)
                state=self.prepare_rewind(sim,state)
                state=self.prepare_next_region(sim,state)
                if self.has_next_region(state):
                    state=self.on_floor_exit(state)
                previous_floor_revealed=state.revealed
                state,clear_reward,msg=sim._advance_floor(state,exit_node,refresh_revealed=False)
                messages.extend(msg)
                if not state.terminal:
                    state=self.initialize_region(state)
                    state=self.initialize_population(state)
                    state=sim._refresh_revealed(state)
                    state=self.on_reveal(state,len(state.revealed-previous_floor_revealed))
                    state=relic_effects.on_floor_enter(self,state)
                    state=self.return_expeditions(state)
                    state=self.initialize_fates(state)
            elif state.resources.action_points<=0:
                if state.portal_context is not None:
                    state=self.return_portal(state)
                    messages.append('黑潭行动力耗尽：返回入口')
            if state.pending_node_id is None and state.resources.action_points<=0 and not state.terminal:
                state,clear_reward,msg=self.chase(sim,state)
                messages.extend(msg)
        previous_revealed=state.revealed
        state=sim._refresh_revealed(state)
        state=self.on_reveal(state,len(state.revealed-previous_revealed))
        # Bank transfers supply external capital; the transfer itself earns no
        # gold reward. Subsequent genuine relic acquisitions retain their reward.
        withdrawn=state.total_bank_withdrawn-before.total_bank_withdrawn
        reward_resources=replace(state.resources,gold=state.resources.gold-withdrawn)
        reward=sim.ruleset.objective.resource_reward(before.resources,reward_resources)+clear_reward
        if action.kind==ActionKind.EQUIP:
            reward-=0.01
        needs=state.pending_node_id is not None and state.floor_map.node(state.pending_node_id).requires_observation
        state=state.with_reward(reward,'；'.join(messages))
        return Transition(state,reward,state.terminal,{'action':action,'messages':tuple(messages),
            'needs_observation':needs,'status':'NEEDS_OBSERVATION' if needs else 'OK',
            'bank_withdrawn':state.total_bank_withdrawn,'bank_balance':state.bank_balance,
            'bank_balance_spent':state.bank_balance_spent,
            'profile':'synthetic-item-economy','ledger_entries':len(state.ledger)-len(before.ledger)})

    def return_portal(self,state):
        from .portal import return_portal_state
        if state.portal_context is None:
            raise ValueError('not currently in a Black Pond')
        state=return_portal_state(state,state.portal_context)
        return replace(state,portal_context=None)

    def has_next_region(self,state):
        if state.floor>=5:
            from .ending_rules import after_floor_clear
            return after_floor_clear(state.floor,inventory=state.inventory).next_floor is not None
        return state.floor_index+1<len(state.maps)

    def first_ending_available(self):
        # A selectable positive secrecy level requires a previous clear; the
        # alternative endings themselves require ending 1. Explicit observations
        # can still override this account-unlock inference.
        return (self.config.first_ending_unlocked if self.config.first_ending_unlocked is not None
                else self.config.difficulty>0)

    def prepare_next_region(self,sim,state):
        """Ending map content is gated by inventory at actual region entry."""
        if not self.has_next_region(state):
            return state
        from .mapgen import MapGenerator
        from .fate_nodes import fate_route_available
        next_index=state.floor_index+1
        if next_index<len(state.maps):
            desired_floor=state.maps[next_index].floor
        else:
            desired_floor=6 if state.floor==5 else None
        if desired_floor not in (5,6):
            return state
        if desired_floor==5:
            enabled=fate_route_available(inventory=state.inventory,first_ending_unlocked=self.first_ending_available())
            target_count=3 if enabled else 0
            if sum(n.node_type==NodeType.STORY for n in state.maps[next_index].nodes)==target_count:
                return state
            config=replace(sim.map_generator.config,enable_second_ending=enabled)
        else:
            if next_index<len(state.maps):
                return state
            config=sim.map_generator.config
        generator=MapGenerator(sim.ruleset,config=config,templates=sim.map_generator.templates,
            map_generation_profile=sim.map_generator.content_total_ranges,
            node_reward_profile=sim.map_generator.node_reward_profile)
        rng,state=self.roll(state,'gated_region:'+str(desired_floor))
        next_map=generator.generate_floor(desired_floor,rng.getrandbits(63))
        maps=list(state.maps)
        if next_index<len(maps):
            maps[next_index]=next_map
        else:
            maps.append(next_map)
        return self.entry(replace(state,maps=tuple(maps)),'ending_map_gate',
            source='account_and_inventory_gated_floor:'+str(desired_floor))

    def initialize_fates(self,state):
        from .fate_nodes import initialize_fates,fate_route_available
        if state.floor!=5 or state.portal_context is not None:
            return replace(state,fate_context=None)
        if not fate_route_available(inventory=state.inventory,first_ending_unlocked=self.first_ending_available()):
            return replace(state,fate_context=None)
        rng,state=self.roll(state,'fate_group')
        context=initialize_fates(state.floor_map,inventory=state.inventory,rng=rng,allow_synthetic=True)
        if context:
            for assumption in context.assumptions:
                state=self.entry(state,'model_assumption',source=assumption)
        return replace(state,fate_context=context)

    def expedition_menu(self,state,node):
        options=[]
        if state.portal_context is None and state.floor in (2,3,4):
            for operator_id in sorted(state.available_formal_operator_ids):
                options.append(EventOption('expedition_inside:'+operator_id,
                    '探索树的内部：'+operator_id,operation='expedition_inside',item_id=operator_id))
                if self.first_ending_available():
                    options.append(EventOption('expedition_source:'+operator_id,
                        '探索树的源头：'+operator_id,operation='expedition_source',item_id=operator_id))
        if not state.formal_operator_ids:
            state=self.entry(state,'unresolved_roster',source='expedition:no_observed_formal_operator')
        options.append(EventOption('rest','不派遣，获得2希望',ResourceDelta(hope=2),operation='event_reward'))
        return self.set_options(state,node,options)

    def return_expeditions(self,state):
        if state.portal_context is not None or state.terminal:
            return state
        due=tuple(x for x in state.expeditions if x.return_floor_index==state.floor_index)
        # Remove pending requests before granting anything, so recursion/replay
        # cannot pay the same returned operator a second time.
        state=replace(state,expeditions=tuple(x for x in state.expeditions if x not in due))
        for request in due:
            state=self.entry(state,'operator_returned',quantity=1,source='expedition:'+request.operator_id)
            if request.kind=='source':
                from .ending_rules import BEACON_ID
                state=self.apply(state,ResourceDelta(hope=2),'expedition_return:source')
                state=self.acquire(state,BEACON_ID,'expedition_source_return')
            elif request.kind=='inside':
                from .operator_economy import promote_formal
                state=promote_formal(self,state,request.operator_id,source='expedition_inside_return',hope_cost=0)
                state=self.entry(state,'unresolved_reward',source='expedition_inside:additional_supplies_unobserved_omitted')
            else:
                raise ValueError('unknown expedition kind')
            if self.config.full_tech:
                state=self.grant(state,'pool:MOVE',source='full_technology:expedition_return')
        return state

    def prepare_rewind(self,sim,state):
        from .rewind import has_pending_remembrance,prepare_remembrance_redirect
        if state.portal_context is None and state.floor==4 and has_pending_remembrance(state):
            rng,state=self.roll(state,'remembrance_new_region')
            replacement=sim.map_generator.generate_floor(4,rng.getrandbits(63))
            state=prepare_remembrance_redirect(state,replacement)
            state=self.entry(state,'region_redirect',source='remembrance:region_IV')
        return state

    def initialize_region(self,state):
        from .utopia_economy import generate_region,RegionState,region_coverage,effect_tier
        rng,state=self.roll(state,'region_generation')
        if self.counter(state,'next_hope_soil'):
            candidates=[n for n in state.floor_map.nodes if n.node_type not in
                (NodeType.START,NodeType.BATTLE_ELITE,NodeType.BATTLE_BOSS)]
            center=rng.choice(candidates)
            region=RegionState(center.node_id,'希望的沃土','增益',
                region_coverage(state.floor_map,center.node_id,self.config.difficulty),
                effect_tier(self.config.difficulty),3 if self.config.difficulty>=12 else 2,
                removable=False,selection_evidence='SYNTHETIC_UNIFORM_NONELITE_SOIL_CENTER')
            state=self.set_counter(state,'next_hope_soil',0)
        else:
            region=generate_region(rng,state.floor_map,self.config.difficulty)
        if region:
            state=self.entry(state,'model_assumption',source='region:'+region.selection_evidence)
        return replace(state,region_state=region)

    def initialize_population(self,state):
        from .residents import initialize_residents
        rng,state=self.roll(state,'resident_initial_positions')
        green=bool(state.portal_context and state.portal_context.variation_id==6)
        observed=state.portal_context.resident_node_ids if green else None
        context=initialize_residents(state.floor_map,
            remembrance=self.counter(state,'remembrance:destination_index')==state.floor_index and self.counter(state,'remembrance:spent')>0,
            green_pond=green,observed_positions=observed,rng=rng,allow_synthetic=True)
        for assumption in context.assumptions:
            state=self.entry(state,'model_assumption',source=assumption)
        return replace(state,resident_context=context)

    def move_population(self,state,target):
        from .residents import move_residents
        context=state.resident_context
        if context is None or not context.markers:
            return state
        # Lock an already occupied arrival; other inhabitants move once.
        # Both collision ordering and direction law remain explicit priors.
        locked=tuple(m for m in context.markers if m.node_id==target)
        moving=replace(context,markers=tuple(m for m in context.markers if m.node_id!=target),
            destroyed_node_ids=context.destroyed_node_ids|{target})
        rng,state=self.roll(state,'resident_movement')
        moved=move_residents(state.floor_map,moving,rng=rng,allow_synthetic=True)
        context=replace(context,markers=locked+moved.markers,assumptions=moved.assumptions)
        state=self.entry(state,'model_assumption',source='residents:lock_arrival_before_other_movements_no_move_into_player')
        return replace(state,resident_context=context)

    def chase_reward_menu(self,state):
        """Refresh fixed loot without completing or re-entering its carrier."""
        context=state.chase_reward_context
        if context is None or not state.chase_reward_pending:
            raise ValueError('no live off-map chase reward context')
        options=context.relic_options+context.part_options+self.reward_recruit_options(state)
        if options:
            return self.set_options(state,context.carrier_node,options)
        if state.resources.parts>state.parts_capacity:
            return self.set_options(state,context.carrier_node,self.navigation_options(state))
        maps=list(state.maps)
        maps[state.floor_index]=replace(state.floor_map,nodes=tuple(
            context.carrier_node if x.node_id==context.carrier_node.node_id else x
            for x in state.floor_map.nodes))
        return replace(state,maps=tuple(maps),pending_node_id=None,chase_reward_context=None)

    def chase(self,sim,state):
        # Chase is not a map-node visit; it never triggers on-node part bonuses.
        if state.chase_reward_pending or state.chase_reward_context is not None:
            raise ValueError('the current chase reward must be resolved before another chase')
        if state.portal_context is not None or state.pending_node_id is not None:
            raise ValueError('chase starts only after the pond and current interaction finish')
        from .domain import ChaseRewardContext
        context=ChaseRewardContext(f'chase:{state.chase_count+1}',state.floor,
            state.battle_count+1,state.floor_map.node(state.current_node_id))
        state=replace(state,chase_count=state.chase_count+1,battle_count=state.battle_count+1,
            chase_reward_pending=True,chase_reward_context=context)
        state=self.apply(state,ResourceDelta(tickets=1),'chase')
        state=relic_effects.on_battle(self,state,perfect=None,chase=True)
        if 'rogue_6_relic_fight_30' not in state.inventory:
            state=self.acquire(state,'rogue_6_relic_fight_30','chase:medal')
        if state.floor in (3,5,6):
            from .progression import award_battle_exp,BOSS_EXP
            state=award_battle_exp(self,state,BOSS_EXP[state.floor-1],source='boss_chase')
            state=self.battle_gold(state,5 if state.floor==3 else 8,'boss_chase')
            # Normal boss loot applies to this off-map fight. The source pool
            # mapping and part presence remain explicitly declared priors.
            state=self.entry(state,'model_assumption',source=
                f'boss_chase:floor={state.floor}:normal_boss_loot_source_mapping;'
                'F3_observations_not_direct_chase_or_F5_F6_measurements')
            state=self.entry(state,'model_assumption',source=
                f'boss_chase:unverified_part_presence_probability={self.config.battle_part_probability}')
            from .recruitment_candidates import make_boss_chase_recruit_group
            rng,state=self.roll(state,'boss_chase_recruit_ticket')
            group=make_boss_chase_recruit_group(rng,group_id=context.occurrence_id+':recruit',
                floor=state.floor,radio_owned='rogue_6_relic_legacy_21' in state.inventory,
                observed_base_count=dict(self.config.observed_battle_recruit_candidate_counts).get('BOSS_CHASE'))
            state=replace(state,pending_recruit_candidate_groups=state.pending_recruit_candidate_groups+(group,),
                recruit_candidate_serial=state.recruit_candidate_serial+1)
            state=self.entry(state,'recruit_candidate_group_granted',instance_id=group.group_id,
                quantity=1,source=group.source+':'+group.base_count_evidence)
            state=self.entry(state,'recruit_candidate_offered',instance_id=group.group_id,
                quantity=len(group.candidate_item_ids),source=f'{group.pool_id}:{group.composition_evidence}:'
                f'{group.sampling_assumption}:n={group.sample_count}'+
                (':CLIENT:battle_extra_recruit_ticket:count=1' if group.radio_owned else ''))
            rng,state=self.roll(state,'boss_chase_part_presence')
            part=rng.random()<self.config.battle_part_probability
            count=2+int('rogue_6_relic_legacy_113' in state.inventory)
            state,relics=self.item_choices(state,'RELIC',count,'boss_chase')
            parts=()
            if part:
                state,parts=self.item_choices(state,'PART',2 if self.config.full_tech else 1,'boss_chase')
            def takes(category,candidates):
                return tuple(EventOption(f'take:{context.occurrence_id}:{category}:{index}:{item_id}',
                    self.catalog[item_id].name,operation='take',item_id=item_id,ends_node=False)
                    for index,item_id in enumerate(candidates))
            state=replace(state,chase_reward_context=replace(context,
                relic_options=takes('RELIC',relics),part_options=takes('PART',parts)))
        else:
            from .recruitment import add_unknown_reserve_opportunities
            state=add_unknown_reserve_opportunities(self,state,'normal_chase',1)
        state=self.chase_reward_menu(state)
        return state,sim.ruleset.objective.chase_penalty,['行动力耗尽：追猎结算，等待战利品与招募券处理']

    def complete_chase(self,sim,state):
        """Advance only after the once-issued chase reward is disposed of."""
        if (not state.chase_reward_pending or state.chase_reward_context is not None
                or state.pending_recruit_ticket_ids or state.pending_recruit_candidate_groups
                or state.resources.parts>state.parts_capacity):
            raise ValueError('chase completion requires all current rewards and overflow resolved')
        state=replace(state,chase_reward_pending=False,pending_node_id=None)
        state=self.prepare_rewind(sim,state)
        state=self.prepare_next_region(sim,state)
        if self.has_next_region(state):
            state=self.on_floor_exit(state)
        placeholder=MapNode('CHASE',0,0,0,NodeType.BATTLE_BOSS,0,
            stage_id='ro6_b_4_b' if state.floor==5 else 'ro6_b_6' if state.floor==6 else None)
        previous_floor_revealed=state.revealed
        state,reward,messages=sim._advance_floor(state,placeholder,chased=True,refresh_revealed=False)
        if not state.terminal:
            state=self.initialize_region(state)
            state=self.initialize_population(state)
            state=sim._refresh_revealed(state)
            state=self.on_reveal(state,len(state.revealed-previous_floor_revealed))
            state=relic_effects.on_floor_enter(self,state)
            state=self.return_expeditions(state)
            state=self.initialize_fates(state)
        return state,reward,messages

    def start(self, state, resources_supplied=False):
        profile = self.catalog.profile_rules
        bank_balance=(self.config.observed_initial_bank_balance if self.config.observed_initial_bank_balance is not None
            else self.config.initial_bank_balance)
        state = replace(state,economy_enabled=True,economy_seed=state.floor_map.seed,
            bank_initial_balance=bank_balance,bank_balance=bank_balance,
            total_bank_withdrawn=0,bank_balance_spent=0,
            parts_capacity=10+(4 if self.config.squad=='multilateral_trade' and self.config.full_tech else 2 if self.config.squad=='multilateral_trade' else 0)-(2 if self.config.difficulty>=7 else 0))
        state=self.entry(state,'bank_initial_balance',quantity=bank_balance,source=(
            'OBSERVED_ACCOUNT_BALANCE' if self.config.observed_initial_bank_balance is not None
            else 'ASSUMED_ACCOUNT_START_BALANCE:not_inferred_from_cumulative_investment'))
        if not resources_supplied:
            hp = 4 if self.config.difficulty >= 10 else 6 if self.config.difficulty >= 1 else 8
            state = replace(state,squad_capacity=6+int(self.config.full_tech),resources=replace(state.resources,hp=hp,max_hp=hp,
                gold=8 + (profile['full_technology_initial_gold_add'] if self.config.full_tech else 0),
                shield=profile['full_technology_initial_shield_add'] if self.config.full_tech else 0,
                action_points=state.resources.action_points + (profile['full_technology_initial_floor_ap_add'] if self.config.full_tech else 0)))
            if self.config.full_tech:
                state = self.grant(state,PREFIX+'G_01',profile['full_technology_initial_seeds'],source='full_technology:initial_seeds')
        state=self.initialize_population(self.initialize_region(state))
        if self.config.previous_cleared_floors is None:
            state=self.entry(state,'unresolved_cross_run_profile',source='starting_reward:previous_progress_unknown_omitted')
            return self.initialize_formal_roster(state)
        from .ending_rules import starting_offer
        rng,state=self.roll(state,'starting_offer')
        offer=starting_offer(previous_cleared_floors=self.config.previous_cleared_floors,
            observed_choice_ids=self.config.observed_starting_choice_ids or None,
            rng=rng,allow_synthetic=True)
        if offer.evidence.startswith('synthetic_'):
            state=self.entry(state,'model_assumption',source=offer.evidence)
        if not offer.choices:
            return self.initialize_formal_roster(state)
        options=tuple(EventOption(spec.choice_id,spec.name,
            ResourceDelta(gold=spec.gold,max_hp=spec.max_hp),operation='starting_reward')
            for spec in offer.choices)
        state=self.set_counter(state,'starting_reward_pending',1)
        return self.set_options(state,state.floor_map.node(state.current_node_id),options)

    def initialize_formal_roster(self,state):
        if self.counter(state,'initial_roster_initialized'):
            return state
        from .operator_economy import MECHANIST_ID,mechanist_hope_cost,recruit_formal
        state=self.set_counter(state,'initial_roster_initialized',1)
        operator_ids=self.config.initial_formal_operator_ids
        if not operator_ids and self.config.abstract_initial_formal_operator_count:
            count=self.config.abstract_initial_formal_operator_count
            operator_ids=((MECHANIST_ID,) if self.config.default_initial_mechanist else ())
            operator_ids+=tuple(f'abstract_free_initial_operator_{i+1}' for i in range(count-len(operator_ids)))
            state=self.entry(state,'model_assumption',quantity=count,
                source='initial_roster:chosen_mechanist_and_free_formal_companions' if self.config.default_initial_mechanist
                else 'initial_roster:free_anonymous_formal_operators')
        for operator_id in operator_ids:
            cost=mechanist_hope_cost(initial=True,full_mastery=self.config.mechanist_full_mastery,
                difficulty=self.config.difficulty) if operator_id==MECHANIST_ID else 0
            if self.config.initial_recruitment_hope_spent is not None:
                cost=0
            state=recruit_formal(self,state,operator_id,source='initial_recruitment',hope_cost=cost)
        if self.config.initial_recruitment_hope_spent is not None:
            if self.config.initial_recruitment_hope_spent>state.resources.hope:
                raise ValueError('initial recruitment hope cost exceeds available hope')
            state=self.apply(state,ResourceDelta(hope=-self.config.initial_recruitment_hope_spent),'initial_recruitment:observed_total')
        if state.floor==1 and not self.counter(state,'first_region_enter_done'):
            state=self.set_counter(state,'first_region_enter_done',1)
            state=relic_effects.on_floor_enter(self,state)
        return state
