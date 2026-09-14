"""Versioned research environments and pre-optimizer rollout quality gates.

Profiles describe experimental assumptions, never measured game distributions.
They change no rewards, policy advice, hidden observations or confirmed prices.
"""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path

from .economy import EconomyConfig
from .mapgen import MapGenerator, MapGeneratorConfig
from .simulator import BlackflowSimulator


PRIOR_FIELDS = frozenset({
    'normal_relic_probability', 'battle_part_probability',
    'seed_part_probability', 'stronghold_relic_probability',
    'synthetic_relic_price_multiplier', 'synthetic_battle_shop_base_relic_slots',
    'synthetic_wave_tail_probability',
})


@dataclass(frozen=True)
class TrainingEnvironmentProfile:
    name: str = 'baseline'
    rationale: str = 'Existing synthetic assumptions; not a real-game calibration.'
    economy_overrides: dict = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError('unsupported training environment profile version')
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('profile name is required')
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError('profile rationale is required')
        if not isinstance(self.economy_overrides, dict):
            raise ValueError('economy_overrides must be an object')
        unknown = set(self.economy_overrides) - PRIOR_FIELDS
        if unknown:
            raise ValueError('unsupported research overrides: ' + ', '.join(sorted(unknown)))
        for name, value in self.economy_overrides.items():
            if type(value) not in (int, float):
                raise ValueError(name + ' must be numeric, not a boolean or string')
        EconomyConfig(**self.economy_overrides)

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig'))) if path else cls()

    def build(self):
        return BlackflowSimulator(map_generator=MapGenerator(config=MapGeneratorConfig(
            allow_synthetic_map_sampling=True, allow_synthetic_event_effects=True,
            allow_synthetic_floor6_contents=True, enable_portal=True,
            enable_expedition=True, include_advanced_nodes=True)),
            economy_config=EconomyConfig(**self.economy_overrides))

    def manifest(self, simulator):
        definition = asdict(self)
        return {**definition, 'status': 'SYNTHETIC_RESEARCH_ASSUMPTIONS',
                'profile_sha256': sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
                'environment_sha256': simulator.environment_sha256,
                'economy_config': asdict(simulator.economy.config),
                'combat_model': simulator.economy.config.combat_assumption,
                'real_game_validated': False}


def rollout_quality(rollout, *, max_coverage_fraction=0.25, max_transition_errors=0):
    """Inspect collected data before optimization; retain every rejected sample.

    Expected missing observations may censor samples. Programming exceptions
    default to stopping the run, instead of masquerading as missing game rules.
    """
    if not 0 <= max_coverage_fraction <= 1:
        raise ValueError('coverage fraction must be in [0,1]')
    if type(max_transition_errors) is not int or max_transition_errors < 0:
        raise ValueError('transition error allowance must be a nonnegative integer')
    total = len(rollout['actions'])
    eligible = int(sum(rollout['learning_mask']))
    errors = sum(row['reason'] == 'simulator_transition_error'
                 for row in rollout['coverage_examples'])
    fraction = (total - eligible) / total if total else 1.0
    reasons = []
    if not eligible:
        reasons.append('no learnable transitions')
    if errors > max_transition_errors:
        reasons.append('unexpected simulator exceptions exceed allowance')
    if fraction > max_coverage_fraction:
        reasons.append('censored transition fraction exceeds allowance')
    return {'passed': not reasons, 'reasons': reasons, 'transitions': total,
            'learning_transitions': eligible, 'censored_fraction': fraction,
            'transition_errors': errors,
            'max_coverage_fraction': max_coverage_fraction,
            'max_transition_errors': max_transition_errors}
