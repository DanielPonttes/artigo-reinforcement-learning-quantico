#!/usr/bin/env python3
"""Run a DQN experiment from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.training.dqn import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-starts", type=int, default=None, help="Override learning starts.")
    parser.add_argument("--epsilon-start", type=float, default=None, help="Override initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Override final epsilon.")
    parser.add_argument("--output-dir", default=None, help="Override output root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.seeds is not None and len(args.seeds) > 0:
        config["seeds"] = args.seeds
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_starts is not None:
        config["learning_starts"] = args.learning_starts
    if args.epsilon_start is not None:
        config["epsilon_start"] = args.epsilon_start
    if args.epsilon_end is not None:
        config["epsilon_end"] = args.epsilon_end

    artifacts = run_experiment(config, output_dir=args.output_dir)
    print(f"results_csv={artifacts.results_csv}")
    print(f"summary_json={artifacts.summary_json}")
    print(f"config_json={artifacts.config_json}")


if __name__ == "__main__":
    main()
