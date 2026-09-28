from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
import json
from pathlib import Path
from typing import Any


DEFAULT_CLIENT_DATA = (
    Path(__file__).resolve().parents[1]
    / "source_data"
    / "roguelike_topic_table_full.json"
)


@dataclass(frozen=True, slots=True)
class ClientDataSummary:
    path: str
    sha256: str
    git_blob_sha: str
    topic_id: str
    node_types: int
    choices: int
    choice_scenes: int
    stages: int
    normal_stages: int
    elite_stages: int
    boss_stages: int
    special_stages: int
    chase_stages: int
    duel_stages: int
    relics: int
    items: int
    zones: int
    scrap_types: int
    move_scraps: int
    goods_scraps: int
    passive_scraps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_client_data(
    path: str | Path | None = None,
    *,
    topic_id: str = "rogue_6",
) -> ClientDataSummary:
    """Validate the one complete client table and report useful coverage.

    The workspace also contains transport fragments (``part_*``/``rem_*``) and
    a truncated ``roguelike_topic_table.json``.  They are intentionally never
    used as fallbacks: accepting a fragment would make rules silently depend on
    invalid JSON.
    """

    source = Path(path) if path is not None else DEFAULT_CLIENT_DATA
    source = source.resolve()
    if source.name != "roguelike_topic_table_full.json":
        raise ValueError(
            "expected the complete roguelike_topic_table_full.json; fragments are unsupported"
        )
    raw = source.read_bytes()
    try:
        root = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 client JSON: {source}") from exc

    try:
        details = root["details"][topic_id]
        module = root["modules"][topic_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"client table does not contain {topic_id}") from exc

    required = ("nodeTypeData", "choices", "choiceScenes", "stages", "relics", "items", "zones")
    for key in required:
        if not isinstance(details.get(key), dict):
            raise ValueError(f"{topic_id}.{key} is missing or not an object")

    stages = details["stages"]
    stage_families = {
        stage_id: stage_id.split("_", 2)[1]
        for stage_id in stages
        if stage_id.startswith("ro6_") and "_" in stage_id
    }
    normal = sum(family == "n" for family in stage_families.values())
    elite = sum(family == "e" for family in stage_families.values())
    boss = sum(family == "b" for family in stage_families.values())
    special = sum(family == "t" for family in stage_families.values())
    chase = sum(family == "c" for family in stage_families.values())
    duel = sum(family == "duel" for family in stage_families.values())
    if normal + elite + boss + special + chase + duel != len(stages):
        raise ValueError(f"{topic_id}.stages contains an unknown stage ID family")
    scrap = module.get("scrap", {})

    return ClientDataSummary(
        path=str(source),
        sha256=sha256(raw).hexdigest(),
        git_blob_sha=sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw
        ).hexdigest(),  # noqa: S324 - Git object identity, not cryptography
        topic_id=topic_id,
        node_types=len(details["nodeTypeData"]),
        choices=len(details["choices"]),
        choice_scenes=len(details["choiceScenes"]),
        stages=len(stages),
        normal_stages=normal,
        elite_stages=elite,
        boss_stages=boss,
        special_stages=special,
        chase_stages=chase,
        duel_stages=duel,
        relics=len(details["relics"]),
        items=len(details["items"]),
        zones=len(details["zones"]),
        scrap_types=len(scrap.get("scrapTypeData", {})),
        move_scraps=len(scrap.get("moveScrapData", {})),
        goods_scraps=len(scrap.get("goodsScrapData", {})),
        passive_scraps=len(scrap.get("passiveScrapData", {})),
    )
