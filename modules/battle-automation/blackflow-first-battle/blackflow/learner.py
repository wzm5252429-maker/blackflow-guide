from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PlanStats:
    q: float = 0.0
    visits: int = 0
    wins: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, float | int]) -> "PlanStats":
        return cls(float(value.get("q", 0.0)), int(value.get("visits", 0)), int(value.get("wins", 0)))

    def to_dict(self) -> dict[str, float | int]:
        return {"q": self.q, "visits": self.visits, "wins": self.wins}


class QLearner:
    """Persistent epsilon-greedy Q learning over legal user-authored plans."""

    def __init__(
        self,
        path: Path,
        alpha: float = 0.25,
        gamma: float = 0.0,
        epsilon: float = 0.2,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.985,
        seed: int | None = None,
    ) -> None:
        self.path = path
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.table: dict[str, dict[str, PlanStats]] = {}
        self.episodes = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.episodes = int(data.get("episodes", 0))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        for context, plans in data.get("table", {}).items():
            self.table[context] = {plan: PlanStats.from_dict(stats) for plan, stats in plans.items()}

    def _stats(self, context: str, plan_id: str) -> PlanStats:
        return self.table.setdefault(context, {}).setdefault(plan_id, PlanStats())

    def select(self, context: str, plan_ids: Iterable[str], training: bool = True) -> str:
        ids = list(plan_ids)
        if not ids:
            raise ValueError("No plans are available.")
        unvisited = [plan_id for plan_id in ids if self._stats(context, plan_id).visits == 0]
        if training and unvisited:
            return self.random.choice(unvisited)
        if training and self.random.random() < self.epsilon:
            return self.random.choice(ids)
        best_q = max(self._stats(context, plan_id).q for plan_id in ids)
        best = [plan_id for plan_id in ids if self._stats(context, plan_id).q == best_q]
        return self.random.choice(best)

    def update(self, context: str, plan_id: str, reward: float, next_max_q: float = 0.0, won: bool = False) -> None:
        stats = self._stats(context, plan_id)
        target = reward + self.gamma * next_max_q
        stats.q += self.alpha * (target - stats.q)
        stats.visits += 1
        stats.wins += int(won)
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "table": {
                context: {plan: stats.to_dict() for plan, stats in plans.items()}
                for context, plans in self.table.items()
            },
        }
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def summary(self, context: str) -> list[dict[str, float | int | str]]:
        rows = []
        for plan, stats in sorted(self.table.get(context, {}).items()):
            rows.append({
                "plan": plan,
                "q": round(stats.q, 4),
                "visits": stats.visits,
                "wins": stats.wins,
                "win_rate": round(stats.wins / stats.visits, 4) if stats.visits else 0.0,
            })
        return rows

