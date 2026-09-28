"""Observed DUEL-middle pairs: two candidates, one physical reward.

The 455 paired observations have a joint distribution.  The larger 1009-menu
aggregate supplies only candidate marginals (2018 occurrences), not joint
pairs; never combine these denominators or sample two independent marginals.
These are empirical frequencies, not server probabilities.  No extra part or
recruitment ticket is granted here, and existing reward groups never reroll.

The pair listing is unordered observation data; its order is not evidence of
the game's left/right presentation order.  Choosing the better item is valid
after the actual pair is revealed.  Expected-max evaluation integrates over
all 455 observed menus, rather than allowing selection of a lucky random pair.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Callable, Mapping


POOL_ID = "rogue_6:狭路相逢：中"
PAIR_EVIDENCE = "EMPIRICAL_JOINT_CANDIDATE_PAIRS_NOT_SERVER_PROBABILITIES"
MIDDLE_ITEM_IDS = frozenset(
    [f"rogue_6_scrap_G_{number:02}" for number in (3, 4, 5, 6, 9, 10)]
    + [f"rogue_6_scrap_M_{number:02}" for number in (7, 8, 9, 10, 11, 12)]
)
EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "evidence" / "rogue6_duel_middle_candidates_v1.json"


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _pair(values):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("a middle reward has exactly two candidate slots")
    if any(not isinstance(value, str) or value not in MIDDLE_ITEM_IDS for value in values):
        raise ValueError("candidate is not in the documented middle pool")
    # Repeated items are not silently deduplicated. No such pair was observed
    # in the pinned sample, but the observations do not establish a server ban.
    return tuple(values)


@dataclass(frozen=True, slots=True)
class WeightedDuelPair:
    item_ids: tuple[str, str]
    occurrence_count: int

    def __post_init__(self):
        object.__setattr__(self, "item_ids", _pair(self.item_ids))
        _positive_integer(self.occurrence_count, "pair occurrence count")


@dataclass(frozen=True, slots=True)
class DuelMiddleDistribution:
    pairs: tuple[WeightedDuelPair, ...]
    paired_sample_count: int
    aggregate_sample_count: int
    aggregate_candidate_count: int
    evidence: str = PAIR_EVIDENCE

    def __post_init__(self):
        object.__setattr__(self, "pairs", tuple(self.pairs))
        _positive_integer(self.paired_sample_count, "paired sample count")
        _positive_integer(self.aggregate_sample_count, "aggregate sample count")
        _positive_integer(self.aggregate_candidate_count, "aggregate candidate count")
        if not self.pairs or not all(isinstance(row, WeightedDuelPair) for row in self.pairs):
            raise ValueError("weighted observed pairs are required")
        if sum(row.occurrence_count for row in self.pairs) != self.paired_sample_count:
            raise ValueError("pair weights do not sum to their own sample count")
        canonical = [tuple(sorted(row.item_ids)) for row in self.pairs]
        if len(canonical) != len(set(canonical)):
            raise ValueError("duplicate pair rows would double-count observation mass")
        if (self.aggregate_sample_count < self.paired_sample_count
                or self.aggregate_candidate_count != 2*self.aggregate_sample_count):
            raise ValueError("aggregate menu/candidate denominators are inconsistent")


def _marginals(record: Mapping, *, expected_ids: frozenset[str]):
    samples = _positive_integer(record.get("sampleCount"), "marginal sample count")
    draws = _positive_integer(record.get("drawCount"), "marginal candidate count")
    if draws != 2*samples or record.get("interpretation") != "observedFrequencyOnly":
        raise ValueError("marginals must describe observed two-candidate menus")
    counts = Counter()
    for row in record.get("members", ()):
        item_id = row.get("itemId")
        if item_id not in expected_ids or item_id in counts:
            raise ValueError("unknown or repeated marginal member")
        count = _positive_integer(row.get("occurrenceCount"), "marginal occurrence count")
        if type(row.get("totalOccurrenceCount")) is not int or row["totalOccurrenceCount"] != draws:
            raise ValueError("member denominator differs from candidate count")
        rate = row.get("occurrenceRate")
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or not math.isclose(rate, count/draws, abs_tol=1e-8)):
            raise ValueError("reported marginal rate does not match its count")
        counts[item_id] = count
    if frozenset(counts) != expected_ids or sum(counts.values()) != draws:
        raise ValueError("marginal member mass is incomplete or inconsistent")
    return samples, draws, counts


def validate_duel_middle_evidence(document: Mapping) -> DuelMiddleDistribution:
    """Validate identity, counts, pair/marginal agreement, and source separation."""
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or document.get("topic_id") != "rogue_6"
            or document.get("node_type") != "DUEL" or document.get("branch") != "middle"):
        raise ValueError("not the supported DUEL-middle observation schema")
    rules = document.get("rules", {})
    for key, expected in (("candidate_count", 2), ("physical_rewards_selected", 1), ("recruit_ticket_count", 2)):
        if type(rules.get(key)) is not int or rules[key] != expected:
            raise ValueError("candidate and actual reward quantities must remain distinct")
    member_ids = document.get("item_ids", ())
    if (not isinstance(member_ids, (list, tuple)) or len(member_ids) != len(MIDDLE_ITEM_IDS)
            or not all(isinstance(item_id, str) for item_id in member_ids)
            or frozenset(member_ids) != MIDDLE_ITEM_IDS):
        raise ValueError("the snapshot contains an unknown or duplicated pool member")
    observed = document.get("paired_observations", {})
    pairs_record = observed.get("pairs", {})
    pairs = tuple(WeightedDuelPair(row.get("itemIds"), row.get("occurrenceCount"))
                  for row in pairs_record.get("members", ()))
    pair_samples, _, pair_marginals = _marginals(observed.get("candidate_marginals", {}),
                                               expected_ids=MIDDLE_ITEM_IDS)
    if type(pairs_record.get("sampleCount")) is not int or pairs_record["sampleCount"] != pair_samples:
        raise ValueError("paired marginals use another source's denominator")
    from_pairs = Counter()
    for row in pairs:
        for item_id in row.item_ids:
            from_pairs[item_id] += row.occurrence_count
    if from_pairs != pair_marginals:
        raise ValueError("joint pairs do not reproduce their source's candidate marginals")
    aggregate_samples, aggregate_draws, aggregate = _marginals(
        document.get("aggregate_candidate_marginals", {}), expected_ids=MIDDLE_ITEM_IDS)
    auto_samples, _, auto = _marginals(document.get("autobattle_candidate_marginals", {}),
                                      expected_ids=MIDDLE_ITEM_IDS)
    if pair_samples + auto_samples != aggregate_samples or pair_marginals + auto != aggregate:
        raise ValueError("aggregate must be paired source plus separate autobattle marginals")
    return DuelMiddleDistribution(pairs, pair_samples, aggregate_samples, aggregate_draws)


@lru_cache(maxsize=1)
def load_duel_middle_distribution() -> DuelMiddleDistribution:
    return validate_duel_middle_evidence(json.loads(EVIDENCE_PATH.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class DuelMiddleReward:
    group_id: str
    candidate_item_ids: tuple[str, str]
    paired_sample_count: int
    evidence: str = PAIR_EVIDENCE
    selected_index: int | None = None
    declined: bool = False

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("a unique real reward group ID is required")
        object.__setattr__(self, "candidate_item_ids", _pair(self.candidate_item_ids))
        _positive_integer(self.paired_sample_count, "paired sample count")
        if self.selected_index is not None and (type(self.selected_index) is not int
                                                or self.selected_index not in (0, 1)):
            raise ValueError("selected index must identify one candidate slot")
        if type(self.declined) is not bool or self.declined and self.selected_index is not None:
            raise ValueError("a reward cannot be selected and declined")

    @property
    def settled(self) -> bool:
        return self.declined or self.selected_index is not None

    @property
    def selected_item_id(self) -> str | None:
        return None if self.selected_index is None else self.candidate_item_ids[self.selected_index]


def make_duel_middle_reward(
    rng, *, group_id: str, existing_group: DuelMiddleReward | None = None,
    distribution: DuelMiddleDistribution | None = None,
) -> DuelMiddleReward:
    """One joint draw per reward; repeat observations use the already rolled pair."""
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("a unique real reward group ID is required")
    if existing_group is not None:
        if existing_group.group_id != group_id:
            raise ValueError("existing reward belongs to another group")
        return existing_group
    distribution = distribution or load_duel_middle_distribution()
    offset = rng.randrange(distribution.paired_sample_count)
    if type(offset) is not int or not 0 <= offset < distribution.paired_sample_count:
        raise ValueError("RNG returned an invalid observation index")
    for pair in distribution.pairs:
        if offset < pair.occurrence_count:
            return DuelMiddleReward(group_id, pair.item_ids, distribution.paired_sample_count,
                                    evidence=distribution.evidence)
        offset -= pair.occurrence_count
    raise AssertionError("validated pair mass unexpectedly exhausted")


def select_duel_middle_candidate(reward: DuelMiddleReward, index: int):
    """Resolve exactly one part; the unchosen slot never becomes an owned item."""
    if reward.settled or type(index) is not int or index not in (0, 1):
        raise ValueError("reward is settled or candidate index is invalid")
    after = replace(reward, selected_index=index)
    return after, after.selected_item_id


def decline_duel_middle_reward(reward: DuelMiddleReward) -> DuelMiddleReward:
    if reward.settled:
        raise ValueError("reward is already settled")
    return replace(reward, declined=True)


def expected_duel_middle_value(
    value_of: Callable[[str], float], *, distribution: DuelMiddleDistribution | None = None,
) -> float:
    """E[max(value(candidate_1), value(candidate_2))] under the paired subset."""
    distribution = distribution or load_duel_middle_distribution()
    values = {item_id: float(value_of(item_id))
              for item_id in {item_id for row in distribution.pairs for item_id in row.item_ids}}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("item values must be finite")
    return sum(row.occurrence_count * max(values[item_id] for item_id in row.item_ids)
               for row in distribution.pairs) / distribution.paired_sample_count
