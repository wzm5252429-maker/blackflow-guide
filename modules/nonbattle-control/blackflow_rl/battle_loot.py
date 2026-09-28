"""Concrete stage chest draws from client level files, without an outer roll.

All 31 ordinary battle layouts contain two independent weighted groups. The
37 other level files do not contain these groups. Bosses, residents, encounters
and pursuits must therefore use their concrete stage IDs, never ordinary loot.
General relic/scrap reward presence is absent from these level definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re


PATH = Path(__file__).resolve().parents[1] / "data/evidence/rogue6_economy_chest_groups_v1.json"
CLIENT_EVIDENCE = "A_CLIENT_LEVEL_FIELDS"
STAGE_SELECTION_PRIOR = "SYNTHETIC_UNIFORM_WITHIN_FLOOR_AND_NODE_TYPE"
OPEN_ALL_ASSUMPTION = "ORACLE_ALL_SPAWNED_CHESTS_DESTROYED_OR_OPENED"


@lru_cache(maxsize=1)
def evidence():
    return json.loads(PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class StageSelection:
    stage_id: str
    name: str
    level_id: str
    selection_evidence: str


@dataclass(frozen=True, slots=True)
class ChestDraw:
    group_id: str
    outcome: str
    trap_id: str
    weight: int
    group_total_weight: int
    evidence: str = CLIENT_EVIDENCE


@dataclass(frozen=True, slots=True)
class ChestLoot:
    gold: int
    scraps: int
    outcome_evidence: str


def stage_candidates(floor: int, node_type) -> tuple[StageSelection, ...]:
    """Ordinary/elite stage membership; selection frequencies remain unknown."""
    if floor not in range(1, 7):
        raise ValueError("ordinary stage floor must be 1..6")
    node_type = getattr(node_type, "name", node_type)
    if node_type not in ("BATTLE_NORMAL", "BATTLE_ELITE"):
        return ()
    letter = "n" if node_type == "BATTLE_NORMAL" else "e"
    pattern = re.compile(rf"ro6_{letter}_{floor}_\d+$")
    return tuple(StageSelection(row["stage_id"], row["name"], row["levelId"],
                                STAGE_SELECTION_PRIOR)
                 for row in evidence()["stage_metadata"] if pattern.fullmatch(row["stage_id"]))


def choose_stage(rng, *, floor: int, node_type) -> StageSelection:
    """Explicit uniform stage prior; preferably use an observed stage ID."""
    candidates = stage_candidates(floor, node_type)
    if not candidates:
        raise ValueError("a concrete observed stage is required for special battles")
    return rng.choice(candidates)


def stage_by_id(stage_id: str) -> StageSelection:
    row = next((row for row in evidence()["stage_metadata"] if row["stage_id"] == stage_id), None)
    if row is None:
        raise ValueError("unknown stage: " + stage_id)
    return StageSelection(stage_id, row["name"], row["levelId"], "OBSERVED_STAGE_ID")


def draw_chests(rng, stage: StageSelection | str) -> tuple[ChestDraw, ...]:
    if isinstance(stage, str):
        stage = stage_by_id(stage)
    groups = evidence()["levels"][stage.level_id]["chest_groups"]
    result = []
    for group_id, rows in groups.items():
        if any(row["outcome"] == "unknown" or row["hiddenGroup"] is not None for row in rows):
            raise ValueError("conditional/unknown chest group needs observed resolution")
        weights = [row["weight"] for row in rows]
        row = rng.choices(rows, weights=weights, k=1)[0]
        result.append(ChestDraw(group_id, row["outcome"], row["key"].split("#")[0],
                                row["weight"], sum(weights)))
    return tuple(result)


def resolve_chests(draws: tuple[ChestDraw, ...], *, collected_groups=None,
                   assume_all_opened: bool = False) -> ChestLoot:
    """Separate spawning from success at collecting rewards.

    A combat victory is not evidence that all chests were collected. Specify
    observed group IDs, or explicitly enable the independent oracle assumption.
    Tree chests cost 50 deployment points; opening is not guaranteed by victory.
    """
    if collected_groups is None and not assume_all_opened:
        raise ValueError("observed chest collection or explicit oracle assumption is required")
    if collected_groups is not None and assume_all_opened:
        raise ValueError("choose observed outcomes or an oracle assumption, not both")
    eligible = set(draw.group_id for draw in draws) if assume_all_opened else set(collected_groups)
    if not eligible.issubset({draw.group_id for draw in draws}):
        raise ValueError("collected group was not drawn")
    gold = sum({"normal": 2, "mutant": 5, "mimic": 10}.get(draw.outcome, 0)
               for draw in draws if draw.group_id in eligible)
    scraps = sum(draw.outcome == "tree" for draw in draws if draw.group_id in eligible)
    return ChestLoot(gold, scraps, OPEN_ALL_ASSUMPTION if assume_all_opened else "OBSERVED_CHEST_COLLECTION")
