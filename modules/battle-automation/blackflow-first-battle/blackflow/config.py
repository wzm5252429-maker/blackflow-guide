from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Strategy file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ConfigError("The strategy root must be a JSON object.")
    _validate(data)
    return ProjectConfig(config_path, data)


def _validate(data: dict[str, Any]) -> None:
    required = ("window", "base_resolution", "plans", "outcome", "learning")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"Missing required fields: {', '.join(missing)}")

    resolution = data["base_resolution"]
    if not (
        isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(v, int) and v > 0 for v in resolution)
    ):
        raise ConfigError("base_resolution must be [width, height] using positive integers.")

    plans = data["plans"]
    if not isinstance(plans, list) or not plans:
        raise ConfigError("plans must contain at least one battle plan.")
    ids: set[str] = set()
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict) or not isinstance(plan.get("id"), str):
            raise ConfigError(f"plans[{index}] must have a string id.")
        if plan["id"] in ids:
            raise ConfigError(f"Duplicate plan id: {plan['id']}")
        ids.add(plan["id"])
        if not isinstance(plan.get("actions"), list):
            raise ConfigError(f"Plan {plan['id']} must have an actions array.")

    learning = data["learning"]
    if learning.get("algorithm", "q_learning") != "q_learning":
        raise ConfigError("This version supports learning.algorithm = q_learning only.")
    for field in ("alpha", "gamma", "epsilon"):
        value = float(learning.get(field, {"alpha": 0.25, "gamma": 0.0, "epsilon": 0.2}[field]))
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"learning.{field} must be between 0 and 1.")

