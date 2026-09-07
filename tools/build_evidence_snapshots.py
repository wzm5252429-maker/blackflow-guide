"""Build compact, reproducible evidence snapshots for the Blackflow rules.

This script deliberately copies only client-authored display metadata.  It
does not infer scene-to-choice bindings or server-side reward execution.
"""

from __future__ import annotations

from hashlib import sha1, sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "source_data" / "roguelike_topic_table_full.json"
MANUAL = ROOT / "黑流树海模拟器规则库.json"
OUT_DIR = ROOT / "data" / "evidence"
CLIENT_OUT = OUT_DIR / "rogue6_client_choice_snapshot_v1.json"
EVENT_OUT = OUT_DIR / "rogue6_noncombat_event_catalog_v1.json"

EXPECTED_CLIENT_SHA256 = (
    "aa2b1fc6ba0cc9ee29b9e6a08803550181c3a27189ac449efbad87608880d35b"
)
EXPECTED_CLIENT_GIT_BLOB = "723f15432e989b6d0d402c38548a74a317f2f97c"
PRTS_REVISION = 422994


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return sha1(header + raw).hexdigest()  # noqa: S324 - Git object identity


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_client_snapshot() -> dict[str, object]:
    raw = CLIENT.read_bytes()
    actual_sha256 = sha256(raw).hexdigest()
    actual_git_blob = _git_blob_sha(raw)
    if actual_sha256 != EXPECTED_CLIENT_SHA256:
        raise RuntimeError(f"unexpected client SHA-256: {actual_sha256}")
    if actual_git_blob != EXPECTED_CLIENT_GIT_BLOB:
        raise RuntimeError(f"unexpected client Git blob SHA: {actual_git_blob}")

    root = json.loads(raw.decode("utf-8"))
    detail = root["details"]["rogue_6"]
    normal_grade_zero = next(
        item
        for item in detail["init"]
        if item["modeId"] == "NORMAL" and item["modeGrade"] == 0
    )

    choice_fields = (
        "id",
        "title",
        "description",
        "lockedCoverDesc",
        "type",
        "leftDecoType",
        "nextSceneId",
        "icon",
        "displayData",
        "forceShowWhenOnlyLeave",
        "isHiddenChoice",
        "sortId",
    )
    scene_fields = (
        "id",
        "title",
        "description",
        "background",
        "titleIcon",
        "subTypeId",
        "useHiddenMusic",
    )
    choices = [
        {field: value.get(field) for field in choice_fields}
        for _key, value in sorted(detail["choices"].items())
    ]
    scenes = [
        {field: value.get(field) for field in scene_fields}
        for _key, value in sorted(detail["choiceScenes"].items())
    ]
    node_types = [
        {"key": key, **value}
        for key, value in sorted(detail["nodeTypeData"].items())
    ]
    return {
        "schema_version": 1,
        "artifact_id": "rogue6-client-choice-snapshot-v1",
        "source": {
            "repository": "Kengxxiao/ArknightsGameData",
            "path": "zh_CN/gamedata/excel/roguelike_topic_table.json",
            "git_blob_sha": actual_git_blob,
            "sha256": actual_sha256,
            "verified_at": "2026-09-01",
        },
        "proves": [
            "node type display metadata",
            "choice id/title/description/type/nextSceneId/display hints",
            "choice scene title and prose",
            "NORMAL grade-0 initial resources",
        ],
        "does_not_prove": [
            "scene-to-choice availability graph",
            "server-side costs or effects",
            "event, reward, shop, or map sampling weights",
        ],
        "counts": {
            "node_types": len(node_types),
            "choices": len(choices),
            "choice_scenes": len(scenes),
        },
        "initial_normal_grade_zero": {
            key: normal_grade_zero[key]
            for key in (
                "modeId",
                "modeGrade",
                "initialHp",
                "initialMaxHp",
                "initialPopulation",
                "initialGold",
                "initialSquadCapacity",
                "initialShield",
            )
        },
        "node_types": node_types,
        "choices": choices,
        "choice_scenes": scenes,
    }


def build_event_catalog() -> dict[str, object]:
    source = json.loads(MANUAL.read_text(encoding="utf-8"))
    events = [
        item for item in source["events"] if item["name"] != "狭路相逢·右档"
    ]
    if len(events) != 38:
        raise RuntimeError(f"expected 38 real event groups, got {len(events)}")
    return {
        "schema_version": 1,
        "artifact_id": "rogue6-noncombat-event-catalog-v1",
        "status": "evidence_catalog_not_server_executable_model",
        "sources": source["sources"],
        "source_pins": {
            "client_sha256": EXPECTED_CLIENT_SHA256,
            "client_git_blob_sha": EXPECTED_CLIENT_GIT_BLOB,
            "prts_event_page_revision": PRTS_REVISION,
            "reviewed_at": "2026-09-01",
        },
        "evidence_levels": source["meta"]["evidence_levels"],
        "runtime_policy": {
            "unknown_probability": "NEEDS_OBSERVATION; never assume uniform",
            "unknown_effect": "NEEDS_OBSERVATION; never substitute a no-op or generic reward",
            "training": "synthetic effects require explicit opt-in and cannot be called verified",
        },
        "coverage": {
            "real_event_groups": len(events),
            "summary_options": sum(len(item["options"]) for item in events),
            "client_choices": 396,
            "client_choice_scenes": 338,
            "note": (
                "The summaries encode publicly documented semantic branches; "
                "the exact client display graph remains in the client snapshot."
            ),
        },
        "execution_order": source["execution_order"],
        "required_state": source["required_state"],
        "node_rules": source["nodes"],
        "events": events,
        "unknowns": source["unknowns"],
    }


def main() -> None:
    _write_json(CLIENT_OUT, build_client_snapshot())
    _write_json(EVENT_OUT, build_event_catalog())
    print(CLIENT_OUT)
    print(EVENT_OUT)


if __name__ == "__main__":
    main()
