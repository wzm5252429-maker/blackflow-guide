"""Integrity checks and evidence boundaries for the Blackflow rule snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLIENT_SOURCE = ROOT / "source_data" / "roguelike_topic_table_full.json"
CLIENT_SNAPSHOT = ROOT / "data" / "evidence" / "rogue6_client_choice_snapshot_v1.json"
EVENT_CATALOG = ROOT / "data" / "evidence" / "rogue6_noncombat_event_catalog_v1.json"
MAP_CONFLICTS = ROOT / "data" / "evidence" / "map_rule_conflicts_v1.json"
FLOOR6_FIXTURE = ROOT / "tests" / "golden" / "lubiao_floor6_v1.json"
MAP_RULE_FIXTURE = ROOT / "tests" / "golden" / "lubiao_map_rules_v1.json"

EXPECTED_ARTIFACT_SHA256 = {
    "rogue6_client_choice_snapshot_v1.json": (
        "c0dc1f4da2d39108da082cf13c1104d80aca468aedb9d24c5855efea932cf8bf"
    ),
    "rogue6_noncombat_event_catalog_v1.json": (
        "9f2d56f5c6881bcb893677a542a70f6d23eb68c3449a5074dcca1b3153edade1"
    ),
    "map_rule_conflicts_v1.json": (
        "2a2f612e99461de4b1ae3f9d1015214ee984bb6a3c42b775689c87ae026816ed"
    ),
    "lubiao_floor6_v1.json": (
        "1b47805ae2f8aca2c31b4ae7cea3924fdbe4be0bc455109f94d3999c397f1db5"
    ),
    "lubiao_map_rules_v1.json": (
        "9ce85774b09014a150657a0873a569c73a3328245eeea49b6abcc63bc2fd119d"
    ),
}


@dataclass(frozen=True, slots=True)
class EvidenceAudit:
    integrity_ok: bool
    verified_training_ready: bool
    client_source_present: bool
    client_sha256: str
    client_git_blob_sha: str
    client_choices: int
    client_choice_scenes: int
    event_groups: int
    map_rule_conflicts: int
    floor6_fixture_sha256: str
    map_rule_count_fixture_sha256: str
    map_rule_distance_fixture_sha256: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required evidence artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence JSON: {path}") from exc


def _git_blob_sha(raw: bytes) -> str:
    return sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()  # noqa: S324


def audit_evidence() -> EvidenceAudit:
    client_snapshot = _load(CLIENT_SNAPSHOT)
    event_catalog = _load(EVENT_CATALOG)
    map_conflicts = _load(MAP_CONFLICTS)
    floor6_fixture = _load(FLOOR6_FIXTURE)
    map_rule_fixture = _load(MAP_RULE_FIXTURE)
    pinned = client_snapshot["source"]
    failures: list[str] = []
    client_source_present = CLIENT_SOURCE.is_file()
    if client_source_present:
        client_raw = CLIENT_SOURCE.read_bytes()
        client_sha = sha256(client_raw).hexdigest()
        client_blob = _git_blob_sha(client_raw)
        if b"\xef\xbf\xbd" in client_raw:
            failures.append("client source contains Unicode replacement characters")
    else:
        # The published decision-core package keeps the compact, pinned extract
        # rather than duplicating the 18 MB upstream table.  This is auditable as
        # an artifact but cannot be re-derived locally until the source is added.
        client_sha = str(pinned["sha256"])
        client_blob = str(pinned["git_blob_sha"])
    for artifact_path in (
        CLIENT_SNAPSHOT,
        EVENT_CATALOG,
        MAP_CONFLICTS,
        FLOOR6_FIXTURE,
        MAP_RULE_FIXTURE,
    ):
        artifact_raw = artifact_path.read_bytes()
        if sha256(artifact_raw).hexdigest() != EXPECTED_ARTIFACT_SHA256[
            artifact_path.name
        ]:
            failures.append(
                f"evidence artifact differs from pinned SHA-256: {artifact_path.name}"
            )
        if "\ufffd" in artifact_raw.decode("utf-8"):
            failures.append(f"evidence artifact contains replacement characters: {artifact_path.name}")
    if client_sha != pinned["sha256"]:
        failures.append("client SHA-256 differs from the pinned snapshot")
    if client_blob != pinned["git_blob_sha"]:
        failures.append("client Git blob SHA differs from the pinned repository object")

    counts = client_snapshot["counts"]
    if counts != {"node_types": 21, "choices": 396, "choice_scenes": 338}:
        failures.append("client snapshot counts differ from the golden counts")
    if len(client_snapshot["choices"]) != counts["choices"]:
        failures.append("client choice snapshot is truncated")
    if len(client_snapshot["choice_scenes"]) != counts["choice_scenes"]:
        failures.append("client scene snapshot is truncated")

    events = event_catalog["events"]
    if len(events) != 38:
        failures.append("event catalog must contain 38 real event groups")
    if any(item["name"] == "狭路相逢·右档" for item in events):
        failures.append("simulator-only duel reward tier leaked into the real event catalog")
    conflicts = map_conflicts["conflicts"]
    if map_conflicts.get("status") != "UNRESOLVED" or len(conflicts) != 26:
        failures.append("map source conflict manifest must preserve all 26 unresolved differences")

    raw_fragment = floor6_fixture["raw_fragment"].encode("utf-8")
    floor6_sha = sha256(raw_fragment).hexdigest()
    if floor6_sha != floor6_fixture["fragment_sha256"]:
        failures.append("floor VI raw source fragment SHA differs from the golden fixture")
    count_sha = sha256(map_rule_fixture["count_fragment"].encode("utf-8")).hexdigest()
    distance_sha = sha256(
        map_rule_fixture["distance_fragment"].encode("utf-8")
    ).hexdigest()
    if count_sha != map_rule_fixture["count_fragment_sha256"]:
        failures.append("raw Lubiao node-count fragment SHA differs from the golden fixture")
    if distance_sha != map_rule_fixture["distance_fragment_sha256"]:
        failures.append("raw Lubiao distance fragment SHA differs from the golden fixture")

    blockers = (
        *((
            "complete client source is not present; compact snapshot cannot be re-derived locally",
        ) if not client_source_present else ()),
        "server scene-to-choice availability and effect execution are not public",
        "102 event summary branches are B-grade PRTS interpretations, not a complete server-executable truth table",
        "event/reward/shop/map sampling weights are not public",
        "Lubiao and Arkrog node constraints have 26 unresolved differences",
        "floor VI has eight source-unspecified content slots",
        "floor VI full-reveal behavior lacks an independent pinned observation",
    )
    return EvidenceAudit(
        integrity_ok=not failures,
        verified_training_ready=False,
        client_source_present=client_source_present,
        client_sha256=client_sha,
        client_git_blob_sha=client_blob,
        client_choices=int(counts["choices"]),
        client_choice_scenes=int(counts["choice_scenes"]),
        event_groups=len(events),
        map_rule_conflicts=len(conflicts),
        floor6_fixture_sha256=floor6_sha,
        map_rule_count_fixture_sha256=count_sha,
        map_rule_distance_fixture_sha256=distance_sha,
        blockers=tuple(failures) + blockers,
    )
