#!/usr/bin/env python3
"""Count trainable parameters for a configured Q-network."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.models.factory import create_q_network
from qrl.utils.params import count_all_parameters, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/tables/parameter_counts.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for config_path in args.configs:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        env = gym.make(str(config["env_id"]))
        input_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)
        env.close()

        model = create_q_network(
            agent_name=str(config["agent"]),
            input_dim=input_dim,
            n_actions=n_actions,
            model_config=dict(config.get("model", {})),
        )

        rows.append(
            {
                "config": config_path,
                "experiment": config.get("experiment", ""),
                "agent": config["agent"],
                "env": config["env_id"],
                "trainable_parameters": count_trainable_parameters(model),
                "all_parameters": count_all_parameters(model),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"parameter_counts={output_path}")


if __name__ == "__main__":
    main()

