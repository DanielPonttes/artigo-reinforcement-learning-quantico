#!/usr/bin/env python3
"""Aggregate training CSVs into compact result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="Results CSV files.")
    parser.add_argument("--window", type=int, default=25, help="Final rolling/summary window.")
    parser.add_argument("--output-csv", default="outputs/tables/results_summary.csv")
    parser.add_argument("--output-tex", default="outputs/tables/results_summary.tex")
    return parser.parse_args()


def _format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _format_agent(agent: str) -> str:
    labels = {
        "dqn_mlp": "DQN-MLP",
        "dqn_vqc": "DQN-VQC",
    }
    return labels.get(agent, agent.replace("_", "\\_"))


def main() -> None:
    args = parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    data = data.sort_values(["agent", "seed", "episode"])

    seed_rows = []
    for (agent, seed), run in data.groupby(["agent", "seed"]):
        final_window = run.tail(args.window)
        seed_rows.append(
            {
                "agent": agent,
                "seed": seed,
                "episodes": int(run["episode"].max() + 1),
                "final_mean_reward": float(final_window["reward"].mean()),
                "best_reward": float(run["reward"].max()),
                "total_steps": int(run["steps"].sum()),
                "total_wall_time_sec": float(run["wall_time_sec"].max()),
            }
        )

    per_seed = pd.DataFrame(seed_rows)
    summary = (
        per_seed.groupby("agent")
        .agg(
            seeds=("seed", "nunique"),
            episodes=("episodes", "max"),
            final_mean_reward_mean=("final_mean_reward", "mean"),
            final_mean_reward_std=("final_mean_reward", "std"),
            best_reward_mean=("best_reward", "mean"),
            total_wall_time_sec_mean=("total_wall_time_sec", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    output_tex = Path(args.output_tex)
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Agent & Seeds & Episodes & Final reward & Best reward & Time (s) \\\\",
        "\\hline",
    ]
    for row in summary.to_dict(orient="records"):
        final_reward = (
            f"{_format_float(row['final_mean_reward_mean'])} $\\pm$ "
            f"{_format_float(row['final_mean_reward_std'])}"
        )
        lines.append(
            f"{_format_agent(str(row['agent']))} & {int(row['seeds'])} & {int(row['episodes'])} & "
            f"{final_reward} & {_format_float(row['best_reward_mean'])} & "
            f"{_format_float(row['total_wall_time_sec_mean'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    output_tex.write_text("\n".join(lines), encoding="utf-8")

    print(f"summary_csv={output_csv}")
    print(f"summary_tex={output_tex}")


if __name__ == "__main__":
    main()
