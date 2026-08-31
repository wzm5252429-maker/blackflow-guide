from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .agents import HeuristicEvaluator
from .client_data import DEFAULT_CLIENT_DATA, validate_client_data
from .mapgen import MapGenerator, MapGeneratorConfig
from .mcts import MCTSConfig, PUCTMCTS
from .simulator import BlackflowSimulator


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _configure_utf8_console() -> None:
    """Keep Chinese CLI output readable under legacy Windows code pages."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (LookupError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackflow-rl",
        description="黑流树海模板地图模拟、单玩家 MCTS 与 policy/value 网络训练",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="校验本地 rogue_6 客户端数据")
    validate.add_argument("--path", type=Path, default=DEFAULT_CLIENT_DATA)

    sample = subparsers.add_parser("sample-map", help="抽取模板并打印一套约束地图")
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--floor", type=int, choices=range(1, 7), default=1)
    sample.add_argument(
        "--ending-route",
        choices=("normal", "second", "third"),
        default="normal",
        help="结局路线；third 会在 I–V 后追加固定拓扑、全揭示的第 VI 层",
    )
    sample.add_argument("--json", action="store_true", dest="as_json")

    simulate = subparsers.add_parser("simulate", help="运行一整局基线策略")
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument(
        "--policy", choices=("random", "heuristic", "mcts"), default="mcts"
    )
    simulate.add_argument("--simulations", type=int, default=32)

    plan = subparsers.add_parser("plan", help="对开局状态给出 MCTS 下一步建议")
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--simulations", type=int, default=64)
    plan.add_argument("--checkpoint", type=Path)

    train = subparsers.add_parser("train", help="planner-guided rollout 训练")
    train.add_argument("--episodes", type=int, default=5)
    train.add_argument("--simulations", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--updates", type=int, default=2)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--max-steps", type=int, default=300)
    train.add_argument("--seed", type=int, default=20260831)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--output", type=Path, default=Path("artifacts/blackflow_policy.pt")
    )

    evaluate = subparsers.add_parser("evaluate", help="用相同随机种子比较策略")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seed-start", type=int, default=10000)
    evaluate.add_argument("--simulations", type=int, default=32)
    evaluate.add_argument("--device", default="cpu")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            print(_json(validate_client_data(args.path).to_dict()))
            return 0

        simulator = BlackflowSimulator()
        if args.command == "sample-map":
            if args.floor == 6 and args.ending_route != "third":
                raise ValueError("第6层仅属于三结局；请同时传入 --ending-route third")
            generator = MapGenerator(
                config=MapGeneratorConfig(
                    enable_second_ending=args.ending_route == "second",
                    enable_third_ending=args.ending_route == "third",
                )
            )
            simulator = BlackflowSimulator(map_generator=generator)
            state = simulator.reset(args.seed)
            floor_map = state.maps[args.floor - 1]
            if args.as_json:
                print(
                    _json(
                        {
                            "floor": floor_map.floor,
                            "seed": floor_map.seed,
                            "fingerprint": floor_map.fingerprint,
                            "start": floor_map.start_node_id,
                            "exits": floor_map.exit_node_ids,
                            "nodes": [
                                {
                                    "id": node.node_id,
                                    "row": node.row,
                                    "col": node.col,
                                    "type": node.node_type.value,
                                    "label": simulator.node_label(node),
                                    "distance": node.distance_from_start,
                                    "event": node.event_name,
                                }
                                for node in floor_map.nodes
                            ],
                            "edges": floor_map.edges,
                        }
                    )
                )
            elif args.floor == 1:
                print(simulator.render_text(state))
            else:
                print(
                    f"第{floor_map.floor}层 seed={floor_map.seed} "
                    f"nodes={len(floor_map.nodes)} edges={len(floor_map.edges)} "
                    f"exits={floor_map.exit_node_ids} fingerprint={floor_map.fingerprint}"
                )
                for node in floor_map.nodes:
                    print(
                        f"{node.node_id:>7} ({node.row},{node.col}) "
                        f"d={node.distance_from_start:<2} {simulator.node_label(node)}"
                    )
            return 0

        if args.command == "simulate":
            from .training import run_episode

            stats = run_episode(
                simulator,
                args.seed,
                policy=args.policy,
                simulations=args.simulations,
            )
            print(_json(asdict(stats)))
            return 0

        if args.command == "plan":
            state = simulator.reset(args.seed)
            trainer = None
            if args.checkpoint:
                from .training import Trainer

                trainer = Trainer.load_checkpoint(simulator, args.checkpoint)
                mcts = trainer.make_mcts(
                    args.seed, training=False, simulations=args.simulations
                )
            else:
                mcts = PUCTMCTS(
                    simulator,
                    HeuristicEvaluator(simulator),
                    MCTSConfig(
                        num_simulations=args.simulations,
                        gamma=simulator.ruleset.objective.discount,
                        reward_scale=simulator.ruleset.objective.value_scale,
                        temperature=0.0,
                        seed=args.seed,
                    ),
                )
            result = mcts.search(
                simulator.belief_state(state), temperature=0.0, add_root_noise=False
            )
            candidates = [
                {
                    "action_id": action_id,
                    "visits": result.visit_counts[action_id],
                    "description": simulator.action_description(state, action_id),
                    "selected": action_id == result.selected_action,
                }
                for action_id in simulator.legal_action_ids(state)
            ]
            candidates.sort(key=lambda item: (-item["visits"], item["action_id"]))
            print(simulator.render_text(state))
            print(_json({"root_value": result.root_value, "candidates": candidates}))
            return 0

        if args.command == "train":
            from .training import Trainer, TrainingConfig

            if args.resume:
                trainer = Trainer.load_checkpoint(
                    simulator, args.resume, device=args.device
                )
            else:
                trainer = Trainer(
                    simulator,
                    TrainingConfig(
                        seed=args.seed,
                        episodes=args.episodes,
                        simulations_per_move=args.simulations,
                        batch_size=args.batch_size,
                        updates_per_episode=args.updates,
                        hidden_dim=args.hidden_dim,
                        device=args.device,
                        max_steps_per_episode=args.max_steps,
                    ),
                )
            for report in trainer.train(args.episodes):
                print(_json(report), flush=True)
            checkpoint = trainer.save_checkpoint(args.output)
            print(_json({"checkpoint": str(checkpoint), "rules_sha256": simulator.ruleset.sha256}))
            return 0

        if args.command == "evaluate":
            from .training import Trainer, evaluate_policies

            trainer = (
                Trainer.load_checkpoint(simulator, args.checkpoint, device=args.device)
                if args.checkpoint
                else None
            )
            seeds = range(args.seed_start, args.seed_start + args.episodes)
            print(
                _json(
                    evaluate_policies(
                        simulator,
                        seeds,
                        trainer=trainer,
                        simulations=args.simulations,
                    )
                )
            )
            return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 1
