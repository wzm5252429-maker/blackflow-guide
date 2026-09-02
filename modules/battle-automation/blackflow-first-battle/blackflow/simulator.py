from __future__ import annotations

import random
from pathlib import Path

from .config import ProjectConfig
from .learner import QLearner


def run_simulation(config: ProjectConfig, episodes: int, seed: int = 7) -> list[dict[str, float | int | str]]:
    data = config.data
    learning = data["learning"]
    simulation = data.get("simulation", {})
    probabilities = simulation.get("plan_win_probability", {})
    missing = [plan["id"] for plan in data["plans"] if plan["id"] not in probabilities]
    if missing:
        raise ValueError(f"Simulation probabilities are missing for: {', '.join(missing)}")
    state_file = config.resolve(simulation.get("state_file", "../learning/simulation_q_table.json"))
    if simulation.get("reset_state", True) and state_file.exists():
        state_file.unlink()
    learner = QLearner(
        state_file,
        alpha=float(learning.get("alpha", 0.25)),
        gamma=float(learning.get("gamma", 0.0)),
        epsilon=float(learning.get("epsilon", 0.2)),
        epsilon_min=float(learning.get("epsilon_min", 0.03)),
        epsilon_decay=float(learning.get("epsilon_decay", 0.985)),
        seed=seed,
    )
    rng = random.Random(seed)
    context = str(data.get("context", {}).get("base", "first_floor_first_battle"))
    plan_ids = [plan["id"] for plan in data["plans"]]
    rewards = data["outcome"].get("rewards", {"victory": 1.0, "defeat": -1.0})
    for episode in range(1, episodes + 1):
        plan = learner.select(context, plan_ids, training=True)
        won = rng.random() < float(probabilities[plan])
        reward = float(rewards["victory"] if won else rewards["defeat"])
        learner.update(context, plan, reward, won=won)
        if episode <= 5 or episode % 10 == 0 or episode == episodes:
            print(f"simulation {episode:03d}: plan={plan:<18} result={'win' if won else 'loss'} epsilon={learner.epsilon:.3f}")
    return learner.summary(context)

