"""Bounded teacher-advice priorities with unit mean within every world."""
from collections import defaultdict
import math
import numpy as np


def known_arrival_payoff(row):
    positive = [row['macros'][i] for i in row['positive']]
    if not positive or len({m.semantic for m in positive}) != 1:
        return 0.
    # Existing macro facts already exclude walking, unrevealed destinations,
    # and random-vehicle landings. No teacher or future value is computed here.
    return max(0., 10.*(positive[0].facts[10]+positive[0].facts[11]))


def world_priority_weights(rows, cap=3.):
    if not math.isfinite(cap) or cap < 0:
        raise ValueError('Priority cap must be finite and nonnegative')
    weights = np.asarray([1.+min(cap, known_arrival_payoff(r)) for r in rows], dtype=np.float32)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['episode']].append(i)
    for indices in groups.values():
        weights[indices] /= weights[indices].mean()
    if not np.isfinite(weights).all():
        raise ValueError('Invalid public priority weights')
    return weights
