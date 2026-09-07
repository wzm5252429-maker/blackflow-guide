"""Remove corrupted legacy prose from the map catalogue without touching graphs."""

from __future__ import annotations

import json
from pathlib import Path


PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "rules"
    / "blackflow_map_templates_v1.json"
)


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    removed = 0
    for template in data["templates"]:
        note = template.get("notes", "")
        if "\ufffd" in note:
            template["notes"] = ""
            removed += 1
    if removed:
        data["notes_status"] = (
            f"{removed} corrupted legacy note strings removed; graph data unchanged"
        )
    PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"removed={removed}")


if __name__ == "__main__":
    main()
