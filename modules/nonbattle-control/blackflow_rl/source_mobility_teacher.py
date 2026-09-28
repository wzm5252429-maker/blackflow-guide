"""Source-inspired mobility portfolio hypothesis; original teacher stays frozen.

Evidence: BV13ft264EbS 00:50–01:10 and 02:00–02:40. The author distinguishes
repeatable ordinary movement from long jumps and shows overbuying failures.
The role discount below is project-authored, not a measured game formula.
Only public item definitions and currently held remaining uses are consulted.
"""
from dataclasses import dataclass
import math

from .route_planner import ObservableRouteEvaluator


@dataclass(frozen=True, slots=True)
class MobilityPortfolioConfig:
    strength: float = 1.0
    cross_role_substitution: float = 0.25

    def __post_init__(self):
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in
                   (self.strength, self.cross_role_substitution)):
            raise ValueError('Portfolio weights must be finite and in [0, 1]')


def mobility_role(definition):
    if definition.random_move:
        return 'random_event'
    if not definition.move_range:
        return 'global_all' if 'ALL' in definition.move_target_types else 'global_targeted'
    return 'ordinary'


class SourceMobilityTeacher(ObservableRouteEvaluator):
    def __init__(self, simulator, config=None, route_config=None, *, portfolio=None):
        super().__init__(simulator, config, route_config)
        self.portfolio = portfolio or MobilityPortfolioConfig()

    def _part_future_value(self, state, definition, *, move_uses=None):
        baseline = super()._part_future_value(state, definition, move_uses=move_uses)
        if (not self.portfolio.strength or definition is None or definition.category != 'MOVE'
                or not self._has_future_navigation(state)):
            return baseline
        uses = min((definition.move_uses or 1) if move_uses is None else move_uses,
                   self._remaining_floors(state) * 3)
        if uses <= 0:
            return baseline
        total = substitutes = 0.0
        role = mobility_role(definition)
        for item in state.item_instances:
            if item.category != 'MOVE':
                continue
            count = max(0, item.uses_remaining or 0)
            total += count
            existing = self._definition(item.item_id)
            # Unknown catalog entries conservatively retain original treatment.
            weight = (1.0 if existing is None or mobility_role(existing) == role
                      else self.portfolio.cross_role_substitution)
            substitutes += count * weight
        extent = max((abs(row) + abs(col) for row, col in definition.move_range),
                     default=max(state.floor_map.width, state.floor_map.height))
        raw = uses * (1.5 + min(extent, 6) * 0.55)
        # Replace only the original charge-scarcity component. AP, concepts,
        # cash generation, capacity legality and actual movement stay unchanged.
        delta = raw * (1 / (1 + substitutes / 3) - 1 / (1 + total / 3))
        return baseline + self.portfolio.strength * delta
