"""Conservative candidate/champion checkpoint promotion gate.

The gate consumes paired evaluation results produced with identical seeds. It
does not train or evaluate models itself, keeping promotion independent from
training and every decision auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Sequence


@dataclass(frozen=True)
class GateConfig:
    minimum_pairs: int = 24
    minimum_reward_delta: float = 0.0
    confidence_z: float = 1.645
    maximum_chase_delta: float = 0.25
    minimum_final_relic_delta: float | None = None
    minimum_candidate_final_relics: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_pairs < 2:
            raise ValueError("minimum_pairs must be at least two")
        for name in ("minimum_reward_delta", "confidence_z", "maximum_chase_delta", "minimum_final_relic_delta", "minimum_candidate_final_relics"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.confidence_z <= 0:
            raise ValueError("confidence_z must be positive")


@dataclass(frozen=True)
class PairedObservation:
    seed: int
    candidate_reward: float
    champion_reward: float
    candidate_chases: float = 0.0
    champion_chases: float = 0.0
    candidate_final_relics: float | None = None
    champion_final_relics: float | None = None
    candidate_completed: bool = True
    champion_completed: bool = True


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    pair_count: int
    mean_reward_delta: float
    reward_standard_error: float
    reward_lower_confidence_bound: float
    mean_chase_delta: float
    mean_final_relic_delta: float | None = None
    final_relic_delta_lower_confidence_bound: float | None = None
    candidate_mean_final_relics: float | None = None


def decide_gate(
    observations: Sequence[PairedObservation],
    config: GateConfig = GateConfig(),
) -> GateDecision:
    if len({item.seed for item in observations}) != len(observations):
        raise ValueError("paired evaluation contains duplicate seeds")
    for item in observations:
        for name in ("candidate_reward", "champion_reward", "candidate_chases", "champion_chases", "candidate_final_relics", "champion_final_relics"):
            value = getattr(item, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"nonfinite paired metric: {name}")
        if not item.candidate_completed or not item.champion_completed:
            raise ValueError("paired evaluation includes incomplete episodes")
    requires_relics = config.minimum_final_relic_delta is not None or config.minimum_candidate_final_relics is not None
    if requires_relics and any(item.candidate_final_relics is None or item.champion_final_relics is None for item in observations):
        raise ValueError("relic promotion requires final RELIC inventory metrics for both policies")
    if len(observations) < config.minimum_pairs:
        return GateDecision(
            passed=False,
            reason=f"insufficient paired episodes: {len(observations)} < {config.minimum_pairs}",
            pair_count=len(observations),
            mean_reward_delta=0.0,
            reward_standard_error=math.inf,
            reward_lower_confidence_bound=-math.inf,
            mean_chase_delta=0.0,
        )

    reward_deltas = [item.candidate_reward - item.champion_reward for item in observations]
    chase_deltas = [item.candidate_chases - item.champion_chases for item in observations]
    reward_mean = statistics.fmean(reward_deltas)
    reward_se = (
        statistics.stdev(reward_deltas) / math.sqrt(len(reward_deltas))
        if len(reward_deltas) > 1
        else 0.0
    )
    lower_bound = reward_mean - config.confidence_z * reward_se
    chase_mean = statistics.fmean(chase_deltas)
    relic_mean = relic_lower = candidate_relic_mean = None
    if observations and all(item.candidate_final_relics is not None and item.champion_final_relics is not None for item in observations):
        deltas = [float(item.candidate_final_relics) - float(item.champion_final_relics) for item in observations]
        relic_mean = statistics.fmean(deltas)
        relic_lower = relic_mean - config.confidence_z * statistics.stdev(deltas) / math.sqrt(len(deltas))
        candidate_relic_mean = statistics.fmean(float(item.candidate_final_relics) for item in observations)

    if lower_bound < config.minimum_reward_delta:
        reason = "reward lower confidence bound did not clear the promotion threshold"
        passed = False
    elif chase_mean > config.maximum_chase_delta:
        reason = "candidate increased chase exposure beyond the safety threshold"
        passed = False
    elif config.minimum_final_relic_delta is not None and relic_lower < config.minimum_final_relic_delta:
        reason = "final relic improvement lower confidence bound did not clear the threshold"
        passed = False
    elif config.minimum_candidate_final_relics is not None and candidate_relic_mean <= config.minimum_candidate_final_relics:
        reason = "candidate final relic mean did not strictly exceed the target"
        passed = False
    else:
        reason = "candidate cleared reward and chase thresholds"
        passed = True
    return GateDecision(
        passed=passed,
        reason=reason,
        pair_count=len(observations),
        mean_reward_delta=reward_mean,
        reward_standard_error=reward_se,
        reward_lower_confidence_bound=lower_bound,
        mean_chase_delta=chase_mean,
        mean_final_relic_delta=relic_mean,
        final_relic_delta_lower_confidence_bound=relic_lower,
        candidate_mean_final_relics=candidate_relic_mean,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    torch = __import__("torch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload.get("environment_sha256"):
        raise ValueError(f"checkpoint has no environment identity: {path}")
    return {"environment_sha256": str(payload["environment_sha256"]),
        "task_objective": payload.get("training_config", {}).get("task_objective", "resource_v1"),
        "format_version": payload.get("format_version"),
        "feature_schema_sha256": payload.get("feature_schema_sha256"),
        "search_objective_sha256": payload.get("search_objective_sha256"),
        "policy_constraints_sha256": payload.get("policy_constraints_sha256"),
        "training_seeds": tuple(payload.get("training_seeds", ()))}


def _validate_relic_gate_inputs(candidate_meta: dict[str, Any], champion_meta: dict[str, Any], paired_payload: Any, config: GateConfig) -> str:
    if candidate_meta["task_objective"] != champion_meta["task_objective"]:
        raise ValueError("candidate/champion task objectives differ")
    if candidate_meta["task_objective"] == "resource_v1":
        return "legacy_weighted_reward_research"
    if candidate_meta["task_objective"] != "relic_collection_v1":
        raise ValueError("unsupported checkpoint task objective")
    if config.minimum_final_relic_delta is None or config.minimum_final_relic_delta < 0:
        raise ValueError("relic checkpoints cannot use the legacy reward-only gate; require an explicit nonnegative final RELIC delta and use scripts.evaluate_relic_economy for target acceptance")
    for key in ("format_version", "feature_schema_sha256", "search_objective_sha256", "policy_constraints_sha256"):
        if not candidate_meta.get(key) or candidate_meta[key] != champion_meta.get(key):
            raise ValueError("relic checkpoint compatibility mismatch: " + key)
    if not isinstance(paired_payload, dict) or not paired_payload.get("complete"):
        raise ValueError("relic promotion requires a completed evaluation manifest")
    rows = paired_payload.get("episodes", ())
    seeds = paired_payload.get("seeds", ())
    if not seeds or len(set(seeds)) != len(seeds) or list(seeds) != [row["seed"] for row in rows]:
        raise ValueError("relic promotion requires a complete unique seed manifest matching every requested pair")
    training_seeds = set(candidate_meta["training_seeds"]) | set(champion_meta["training_seeds"])
    for row in rows:
        if row["seed"] in training_seeds:
            raise ValueError("paired evaluation seed overlaps checkpoint training seeds")
        for side in ("candidate", "champion"):
            result = row[side]
            if not result.get("completed") or not result.get("ledger_valid") or not result.get("scope_compliant") or "final_relics" not in result:
                raise ValueError("relic promotion requires explicit completion, first-ending scope, final RELIC inventory and valid item/gold ledgers for every pair")
            count = result["final_relics"]
            if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0 or int(count) != count:
                raise ValueError("final RELIC inventory must be a nonnegative integer count")
    return "paired_relic_candidate_research_not_target_acceptance"


def _load_observations(path: Path) -> list[PairedObservation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("paired results must be a list or an object containing 'episodes'")
    result: list[PairedObservation] = []
    seen_seeds: set[int] = set()
    for row in rows:
        candidate = row["candidate"]
        champion = row["champion"]
        observation = PairedObservation(
            seed=int(row["seed"]),
            candidate_reward=float(candidate["reward"]),
            champion_reward=float(champion["reward"]),
            candidate_chases=float(candidate.get("chases", 0.0)),
            champion_chases=float(champion.get("chases", 0.0)),
            candidate_final_relics=float(candidate["final_relics"]) if "final_relics" in candidate else None,
            champion_final_relics=float(champion["final_relics"]) if "final_relics" in champion else None,
            candidate_completed=bool(candidate.get("completed", True)),
            champion_completed=bool(champion.get("completed", True)),
        )
        if observation.seed in seen_seeds:
            raise ValueError(f"duplicate paired seed: {observation.seed}")
        seen_seeds.add(observation.seed)
        result.append(observation)
    return result


def gate_and_maybe_promote(
    candidate: Path,
    champion: Path,
    paired_results: Path,
    audit_path: Path,
    config: GateConfig,
    promote_to: Path | None = None,
) -> GateDecision:
    candidate_meta = _checkpoint_metadata(candidate)
    champion_meta = _checkpoint_metadata(champion)
    candidate_environment = candidate_meta["environment_sha256"]
    champion_environment = champion_meta["environment_sha256"]
    if candidate_environment != champion_environment:
        raise ValueError("candidate and champion checkpoints use different simulator environments")

    candidate_sha = _file_sha256(candidate)
    champion_sha = _file_sha256(champion)
    paired_payload = json.loads(paired_results.read_text(encoding="utf-8"))
    decision_scope = _validate_relic_gate_inputs(candidate_meta, champion_meta, paired_payload, config)
    provenance = paired_payload.get("provenance") if isinstance(paired_payload, dict) else None
    provenance_valid = False
    if provenance is not None:
        expected = {"candidate_sha256": candidate_sha, "champion_sha256": champion_sha,
            "environment_sha256": candidate_environment}
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("paired result provenance does not match the evaluated checkpoints/environment")
        provenance_valid = True
    if promote_to is not None and not provenance_valid:
        raise ValueError("promotion requires paired result provenance with exact candidate/champion SHA-256 and environment SHA-256")

    decision = decide_gate(_load_observations(paired_results), config)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "candidate": {"path": str(candidate), "sha256": candidate_sha},
        "champion": {"path": str(champion), "sha256": champion_sha},
        "environment_sha256": candidate_environment,
        "config": asdict(config),
        "decision": asdict(decision),
        "paired_results": str(paired_results),
        "paired_result_provenance_valid": provenance_valid,
        "decision_scope": decision_scope,
        "acceptance_passed": False,
        "target_acceptance_tool": "scripts.evaluate_relic_economy",
        "promoted_to": None,
    }
    if decision.passed and promote_to is not None:
        promote_to.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=promote_to.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copy2(candidate, temporary_path)
            temporary_path.replace(promote_to)
        finally:
            temporary_path.unlink(missing_ok=True)
        audit["promoted_to"] = str(promote_to)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research checkpoint promotion gate; target acceptance requires scripts.evaluate_relic_economy")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=24)
    parser.add_argument("--minimum-reward-delta", type=float, default=0.0)
    parser.add_argument("--confidence-z", type=float, default=1.645)
    parser.add_argument("--maximum-chase-delta", type=float, default=0.25)
    parser.add_argument("--minimum-final-relic-delta", type=float)
    parser.add_argument("--minimum-candidate-final-relics", type=float)
    args = parser.parse_args(argv)
    config = GateConfig(
        minimum_pairs=args.minimum_pairs,
        minimum_reward_delta=args.minimum_reward_delta,
        confidence_z=args.confidence_z,
        maximum_chase_delta=args.maximum_chase_delta,
        minimum_final_relic_delta=args.minimum_final_relic_delta,
        minimum_candidate_final_relics=args.minimum_candidate_final_relics,
    )
    decision = gate_and_maybe_promote(
        args.candidate,
        args.champion,
        args.paired_results,
        args.audit,
        config,
        args.promote_to,
    )
    print(json.dumps({**asdict(decision), "acceptance_passed": False,
        "target_acceptance_tool": "scripts.evaluate_relic_economy"}, ensure_ascii=False, indent=2))
    return 0 if decision.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
