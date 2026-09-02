from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EpisodeLogger:
    def __init__(self, root: Path) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = root / stamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self._sequence = 0

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def screenshot(self, label: str, frame: np.ndarray) -> Path:
        self._sequence += 1
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        path = self.directory / f"{self._sequence:04d}_{safe_label}.png"
        cv2.imwrite(str(path), frame)
        return path

