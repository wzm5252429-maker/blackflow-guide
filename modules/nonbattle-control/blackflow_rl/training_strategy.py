"""Replay and optimisation policies shared by the trainer.

The replay sampler separates recent experience from the archive. This keeps a
large replay pool from hiding newly discovered trajectories while retaining old
experience to limit forgetting.
"""

from __future__ import annotations

from collections import deque
import math
import random
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar, overload

import numpy as np


SampleT = TypeVar("SampleT")


class StratifiedReplayBuffer(Generic[SampleT]):
    """A bounded replay buffer with recent/archive priority sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self._samples: deque[SampleT] = deque(maxlen=int(capacity))

    @property
    def maxlen(self) -> int:
        return int(self._samples.maxlen or 0)

    def append(self, sample: SampleT) -> None:
        self._samples.append(sample)

    def extend(self, samples: Iterable[SampleT]) -> None:
        self._samples.extend(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return bool(self._samples)

    def __iter__(self) -> Iterator[SampleT]:
        return iter(self._samples)

    @overload
    def __getitem__(self, index: int) -> SampleT: ...

    @overload
    def __getitem__(self, index: slice) -> list[SampleT]: ...

    def __getitem__(self, index: int | slice) -> SampleT | list[SampleT]:
        return list(self._samples)[index]

    @staticmethod
    def _priority(sample: SampleT, alpha: float) -> float:
        value_target = abs(float(getattr(sample, "value_target", 0.0)))
        policy = np.asarray(getattr(sample, "policy", ()), dtype=np.float64)
        positive = policy[policy > 0]
        if positive.size > 1:
            entropy = float(-(positive * np.log(positive)).sum())
            entropy /= math.log(float(positive.size))
        else:
            entropy = 0.0
        return max(1e-6, 0.25 + value_target + entropy) ** max(0.0, alpha)

    @classmethod
    def _weighted_subset(
        cls,
        population: Sequence[int] | range,
        count: int,
        samples: Sequence[SampleT],
        alpha: float,
        candidate_multiplier: int,
        rng: Any,
    ) -> list[int]:
        if count <= 0 or not population:
            return []
        if len(population) <= count:
            return list(population)

        candidate_count = min(
            len(population),
            max(count, count * max(1, int(candidate_multiplier)), 128),
        )
        candidates = rng.sample(population, candidate_count)
        keyed: list[tuple[float, int]] = []
        for index in candidates:
            weight = cls._priority(samples[index], alpha)
            # Efraimidis-Spirakis weighted sampling without replacement.
            key = rng.random() ** (1.0 / weight)
            keyed.append((key, index))
        keyed.sort(reverse=True)
        return [index for _, index in keyed[:count]]

    def sample(
        self,
        batch_size: int,
        *,
        recent_fraction: float,
        recent_window: int,
        priority_alpha: float,
        candidate_multiplier: int = 8,
        rng: Any = random,
    ) -> list[SampleT]:
        values = list(self._samples)
        count = min(max(0, int(batch_size)), len(values))
        if count == 0:
            return []

        window = min(len(values), max(1, int(recent_window)))
        recent_start = len(values) - window
        desired_recent = min(
            window,
            int(round(count * min(1.0, max(0.0, recent_fraction)))),
        )
        desired_archive = min(recent_start, count - desired_recent)
        desired_recent = min(window, count - desired_archive)

        archive_indices = self._weighted_subset(
            range(0, recent_start),
            desired_archive,
            values,
            priority_alpha,
            candidate_multiplier,
            rng,
        )
        recent_indices = self._weighted_subset(
            range(recent_start, len(values)),
            desired_recent,
            values,
            priority_alpha,
            candidate_multiplier,
            rng,
        )
        indices = archive_indices + recent_indices
        rng.shuffle(indices)
        return [values[index] for index in indices]


def scheduled_learning_rate(
    initial_learning_rate: float,
    minimum_learning_rate: float,
    episodes_completed: int,
    decay_episodes: int,
    decay_rate: float,
) -> float:
    """Return a staircase exponential learning rate with a hard floor."""

    if decay_episodes <= 0:
        return float(initial_learning_rate)
    exponent = max(0, int(episodes_completed)) // int(decay_episodes)
    decayed = float(initial_learning_rate) * float(decay_rate) ** exponent
    return max(float(minimum_learning_rate), decayed)
