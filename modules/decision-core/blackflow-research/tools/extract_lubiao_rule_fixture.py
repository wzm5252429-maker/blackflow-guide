"""Extract pinned node-count/distance rule fragments from the Lubiao bundle."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile


EXPECTED_ASSET_SHA256 = (
    "c81eca3fb71af540ba8184ad3200cd253f70e36a71a98c7949e9cdcacf78c9b0"
)
DEFAULT_ASSET = (
    Path(tempfile.gettempdir())
    / "codex-blackstream-assets"
    / "blackstream-route.js"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "golden"
    / "lubiao_map_rules_v1.json"
)


def _array_assignment(text: str, marker: str) -> str:
    start = text.index(marker)
    opening = text.index("[", start)
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated array assignment: {marker}")


def _split_values(text: str) -> list[str]:
    values: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            values.append(text[start:index])
            start = index + 1
    values.append(text[start:])
    return [item.strip() for item in values]


def _parse_value(value: str, function: str) -> object:
    if value == "null":
        return None
    if value == "Fl":
        return {"kind": "unknown"}
    match = re.fullmatch(rf"{function}\((\d+)(?:,(\d+|null))?\)", value)
    if not match:
        if function == "S":
            match = re.fullmatch(r"tn\((\d+),(\d+)\)", value)
            if match:
                return {
                    "kind": "set",
                    "values": [int(match.group(1)), int(match.group(2))],
                }
        raise ValueError(f"unsupported source rule expression: {value}")
    minimum = int(match.group(1))
    maximum_text = match.group(2)
    maximum = (
        minimum
        if maximum_text is None
        else None
        if maximum_text == "null"
        else int(maximum_text)
    )
    return {"kind": "range", "minimum": minimum, "maximum": maximum}


def _parse_table(fragment: str, function: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r'\{label:"(?P<label>[^"]+)"'
        r'(?:,nodeType:"(?P<node_type>[^"]+)")?'
        r',values:\[(?P<values>.*?)\]\}'
    )
    result: list[dict[str, object]] = []
    for match in pattern.finditer(fragment):
        result.append(
            {
                "label": match.group("label"),
                "node_type": match.group("node_type"),
                "values": [
                    _parse_value(item, function)
                    for item in _split_values(match.group("values"))
                ],
            }
        )
    if len(result) != 25:
        raise ValueError(f"expected 25 rows in {function} table, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = args.asset.read_bytes()
    asset_sha = sha256(raw).hexdigest()
    if asset_sha != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"unexpected Lubiao asset SHA-256: {asset_sha}")
    text = raw.decode("utf-8")
    count_fragment = _array_assignment(text, "ds=[")
    distance_fragment = _array_assignment(text, "ps=[")
    output = {
        "artifact_id": "lubiao-map-rules-2026-08-31",
        "asset_url": "https://www.lubiao.wiki/_nuxt/BmJjpA_n.js",
        "asset_sha256": asset_sha,
        "asset_bytes": len(raw),
        "rules_version": "2026-08-31",
        "count_fragment_sha256": sha256(count_fragment.encode()).hexdigest(),
        "distance_fragment_sha256": sha256(distance_fragment.encode()).hexdigest(),
        "count_fragment": count_fragment,
        "distance_fragment": distance_fragment,
        "counts": _parse_table(count_fragment, "S"),
        "distances": _parse_table(distance_fragment, "y"),
        "distance_indices": [
            "I",
            "II",
            "III",
            "IV-standard",
            "IV-remembrance",
            "V",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)


if __name__ == "__main__":
    main()
