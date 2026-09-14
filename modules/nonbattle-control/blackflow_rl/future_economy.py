"""Public canonical-floor policy estimates; no transitions or hidden maps."""
from __future__ import annotations
import math

from .domain import NodeType


class FutureEconomyValues:
    def _public_future_main_floors(self, state):
        # First ending's ordinary sequence only. No state.maps[next], generated
        # topology, latent rewards, optional portal, or uncommitted rewind.
        return tuple(range(state.floor+1, 6)) if not state.terminal else ()

    def _future_gold_unit_value(self, floor):
        fraction=(floor-1)/4
        return self.config.early_gold_value*(1-fraction)+self.config.late_gold_value*fraction

    def _future_vehicle_quality(self, definition, kind):
        if definition is None or definition.category!='MOVE':
            return 0.0
        targets=definition.move_target_types
        battle=kind==NodeType.BATTLE_NORMAL
        allowed=('ALL' in targets or 'EVENTS' in targets and not battle or kind.value in targets)
        if battle:
            allowed=allowed or any(name.startswith('BATTLE_') and name!='BATTLE_SHOP' for name in targets)
        if not allowed:
            return 0.0
        if definition.random_move:
            return 1.0 if not battle else 0.0
        if not definition.move_range:
            return 1.0
        extent=max(abs(row)+abs(col) for row,col in definition.move_range)
        return min(1.0,0.25+extent/6)

    def _future_vehicle_value(self, state, definition, future_floors):
        """Marginal vehicle value from rules and surviving inventory only."""
        if not future_floors or definition is None or definition.category!='MOVE':
            return 0.0
        survivors=[item for item in state.item_instances if item.category=='MOVE'
            and item.uses_remaining and self._definition(item.item_id) is not None
            and not self._definition(item.item_id).expires_on_floor_change]
        existing_uses=sum(item.uses_remaining for item in survivors)
        uses=min(definition.move_uses or 1,len(future_floors)*3)
        extent=max((abs(row)+abs(col) for row,col in definition.move_range),default=6)
        value=uses*(1.5+min(extent,6)*0.55)/(1+existing_uses/3)
        if definition.move_ap==0:
            value+=uses*self.config.action_point_value
        elif 'rogue_6_relic_cargo_6' in state.inventory:
            value+=self.config.action_point_value
        if definition.item_id.endswith('M_12'):
            value+=3*self.config.action_point_value
        if definition.item_id.endswith('M_10'):
            value+=4*uses*self._future_gold_unit_value(future_floors[0])
        access=getattr(self.config,'future_target_access_probability',0.6)
        activation=0.0
        for suffix,kind,multiplier in (('P_01',NodeType.BATTLE_NORMAL,1),
                ('P_02',NodeType.EMPTY,1),('P_05',NodeType.WISH,1),('P_06',NodeType.SACRIFICE,2)):
            copies=sum(item.item_id.endswith(suffix) for item in state.item_instances)
            if not copies:
                continue
            persistent=suffix in ('P_05','P_06')
            opportunities=(sum(sum(self.simulator.ruleset.node_rules[kind].count_range(floor))/2
                for floor in future_floors)*access) if persistent else 1.0
            occupied=sum(item.uses_remaining*self._future_vehicle_quality(self._definition(item.item_id),kind) for item in survivors)
            quality=self._future_vehicle_quality(definition,kind)
            marginal=max(0.0,min(opportunities,occupied+uses*quality)-min(opportunities,occupied))
            weight=(self.config.one_shot_concept_value_weight if not persistent and len(future_floors)>2 else 1.0)
            activation+=marginal*copies*multiplier*self.config.relic_value*weight
        return value+self.config.activation_mobility_weight*activation

    def _exit_vehicle_value(self, state, node):
        """Call only for committing advance; visiting/staying earns nothing."""
        if (state.portal_context is not None or state.floor>=5 or
                node.node_type not in (NodeType.FINAL,NodeType.EVACUATE) or
                node.node_id not in state.revealed and node.node_id!=state.pending_node_id):
            return 0.0
        futures=self._public_future_main_floors(state)
        ids=(8,9,10,11,12) if node.node_type==NodeType.EVACUATE else (1,2,3,5,6)
        # These are the engine's existing fixed output sets and its explicit
        # uniform sampling prior, not a claim of measured server frequencies.
        value=sum(self._future_vehicle_value(state,self._definition(f'rogue_6_scrap_M_{index:02d}'),futures)
            for index in ids)/len(ids)
        return getattr(self.config,'exit_vehicle_value_weight',1.0)*value

    def _relic_future_bonus(self, state, definition):
        """Finite future opportunities only; no newly acquired relic count."""
        if definition is None:
            return 0.0
        futures=self._public_future_main_floors(state)
        if definition.item_id=='rogue_6_relic_cargo_4':
            distribution=self._pool_distribution(state,'pool_scrap_7')
            reach=getattr(self.config,'marked_future_reach_probability',0.6)
            sell=getattr(self.config,'marked_future_liquidation_probability',0.7)
            total=0.0
            for index,floor in enumerate(futures):
                remaining=len(futures)-index
                for item_id,probability in distribution.items():
                    item=self._definition(item_id)
                    if item is None or item.category!='GOODS':
                        continue
                    price=float(item.sell_price or 0)
                    if item_id.endswith('G_09'):
                        price+=4*min(30,remaining*self.config.vine_future_parts_per_floor)
                    total+=reach*sell*probability*price*self._future_gold_unit_value(floor)
            return total
        if definition.item_id=='rogue_6_relic_artifact_1':
            reward='rogue_6_relic_artifact_2'
            counters=dict(state.event_counters)
            key='relic_layer:rogue_6_relic_artifact_1'
            if counters.get(key+':paid:'+reward,0) or reward in state.inventory:
                return 0.0
            needed=max(0,2-counters.get(key,0))
            if needed==0:
                return self.config.relic_value
            # Only publicly identified shops contribute to the current-area
            # estimate. The already completed entry that sold artifact1 does
            # not retroactively count; re-entry still needs real AP/actions.
            known_shops=sum(node.node_id in state.revealed and node.node_type in (NodeType.BATTLE_SHOP,NodeType.SCRAP_SHOP)
                for node in state.floor_map.nodes)
            current=min(known_shops,max(0,state.resources.action_points-1))*0.5
            future=sum(sum(self.simulator.ruleset.node_rules[kind].count_range(floor))/2
                for floor in futures for kind in (NodeType.BATTLE_SHOP,NodeType.SCRAP_SHOP))
            visits=current+future*getattr(self.config,'future_shop_access_probability',0.6)
            # Explicit policy count approximation, not server probability.
            probability=1-math.exp(-visits)*(1 if needed==1 else 1+visits)
            return self.config.relic_value*probability
        return 0.0
