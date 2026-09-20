"""Explicit video-inspired wave priors, never asserted server probabilities.

BV11cgC6oEix 00:30-01:25 reports integer changes in -8..11, with
the central -6..8 band approximately uniform. Its rare-tail percentages
and verbal ratio are inconsistent, so tail mass is a sensitivity parameter.
Independent draws across owned waves are also a modeling assumption.
"""
from functools import lru_cache
import math

import numpy as np

CORE_DELTAS = tuple(range(-6, 9))
TAIL_DELTAS = (-8, -7, 9, 10, 11)
OBSERVED_DELTAS = frozenset(range(-8, 12))


def validate_tail_probability(value):
    if value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('synthetic wave tail probability must be None or a number in 0..1')


@lru_cache(maxsize=16)
def delta_distribution(tail_probability):
    validate_tail_probability(tail_probability)
    if tail_probability is None:
        return ()
    return tuple((delta, (1-tail_probability)/len(CORE_DELTAS)) for delta in CORE_DELTAS
                 if tail_probability < 1) + tuple((delta, tail_probability/len(TAIL_DELTAS))
                 for delta in TAIL_DELTAS if tail_probability > 0)


def sample_delta(rng, tail_probability):
    distribution = delta_distribution(tail_probability)
    if not distribution:
        raise ValueError('wave outcome needs observation or an explicit research prior')
    return rng.choices([d for d, _ in distribution], weights=[p for _, p in distribution], k=1)[0]


@lru_cache(maxsize=192)
def _expected_values(moves, tail_probability):
    values = np.arange(1000, dtype=float)
    if not moves or tail_probability is None:
        return values
    previous = _expected_values(moves-1, tail_probability)
    return sum(probability * previous[np.clip(values.astype(int)+delta, 0, 999)]
               for delta, probability in delta_distribution(tail_probability))


def expected_appraisal(appraisal, moves, tail_probability):
    validate_tail_probability(tail_probability)
    if type(appraisal) is not int or not 0 <= appraisal <= 999:
        raise ValueError('wave appraisal must be an integer in 0..999')
    if type(moves) is not int or not 0 <= moves <= 80:
        raise ValueError('wave valuation horizon must be an integer in 0..80')
    return float(_expected_values(moves, tail_probability)[appraisal])
