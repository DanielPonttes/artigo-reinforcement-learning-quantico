#!/usr/bin/env python3
"""Plot learning curves from one or more results CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="Results CSV files.")
    parser.add_argument("--window", type=int, default=25, help="Rolling mean window.")
    parser.add_argument("--output", default="outputs/figures/learning_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.csv]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["agent", "seed", "episode"])
    data["rolling_reward"] = data.groupby(["agent", "seed"])["reward"].transform(
        lambda series: series.rolling(args.window, min_periods=1).mean()
    )

    grouped = (
        data.groupby(["agent", "episode"])["rolling_reward"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for agent, agent_data in grouped.groupby("agent"):
        x = agent_data["episode"].to_numpy()
        mean = agent_data["mean"].to_numpy()
        std = agent_data["std"].to_numpy()
        ax.plot(x, mean, label=agent)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward rolling mean (window={args.window})")
    ax.set_title("CartPole learning curves")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"figure={output_path}")


if __name__ == "__main__":
    main()

