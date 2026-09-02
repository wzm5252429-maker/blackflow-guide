from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .simulator import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BlackFlow first-battle closed-loop agent")
    parser.add_argument("--config", required=True, help="Path to the JSON strategy")
    parser.add_argument("--episodes", type=int, default=1, help="Number of real or simulated battles")
    parser.add_argument("--simulate", action="store_true", help="Run the learning loop without controlling the game")
    parser.add_argument("--no-learning", action="store_true", help="Execute the best known plan without updating Q values")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        print("--episodes must be at least 1", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        if args.simulate:
            rows = run_simulation(config, args.episodes)
            print("\nLearned plan values:")
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return 0
        # Keep Windows input dependencies out of simulation-only runs.
        from .controller import EmergencyStop
        from .engine import BattleEngine, EngineError
        from .vision import VisionError
        from .window import WindowError

        engine = BattleEngine(config, training=not args.no_learning)
        engine.run(args.episodes)
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Real mode defines its platform-specific error classes only after lazy import.
        if exc.__class__.__name__ == "EmergencyStop":
            print(f"Stopped safely: {exc}", file=sys.stderr)
            return 130
        if exc.__class__.__name__ in {"EngineError", "VisionError", "WindowError"}:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
