#!/usr/bin/env python3
"""Build the final fuzzy/quantum scenario table with explicit row labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


GROUPS = [
    {
        "label": "Random",
        "params": 0,
        "method": "Reference",
        "json": ["outputs/results/random_policy_20260607T180958Z_summary.json"],
        "metric": "final_20_mean_reward",
    },
    {
        "label": "Fuzzy teacher",
        "params": 0,
        "method": "Reference",
        "json": ["outputs/results/heuristic_policy_20260607T180958Z_summary.json"],
        "metric": "final_20_mean_reward",
    },
    {
        "label": "Raw DQN-MLP",
        "params": 23,
        "method": "Reward RL",
        "json": ["outputs/results/cartpole_mlp_l1_eval100_20260607T181005Z_summary.json"],
        "metric": "greedy_eval_reward_mean",
    },
    {
        "label": "Raw DQN-VQC L1",
        "params": 22,
        "method": "Reward RL",
        "json": ["outputs/results/cartpole_vqc_l1_eval100_20260607T181047Z_summary.json"],
        "metric": "greedy_eval_reward_mean",
    },
    {
            "label": "Distill MLP mem.",
        "params": 23,
        "method": "Fuzzy distill",
        "json": [
            "outputs/results/fuzzy_distill_dqn_mlp_seed0_20260607T191753Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_mlp_seed1_20260607T191754Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_mlp_seed2_20260607T191754Z_summary.json",
        ],
        "metric": "eval_reward_mean",
    },
    {
            "label": "Distill VQC L1 raw",
        "params": 22,
        "method": "Fuzzy distill",
        "json": [
            "outputs/results/fuzzy_distill_dqn_vqc_seed0_20260607T185059Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_vqc_seed1_20260607T184750Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_vqc_seed2_20260607T184925Z_summary.json",
        ],
        "metric": "eval_reward_mean",
    },
    {
            "label": "Distill VQC L1 mem.",
        "params": 22,
        "method": "Fuzzy distill",
        "json": [
            "outputs/results/fuzzy_distill_dqn_vqc_seed0_20260607T192052Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_vqc_seed1_20260607T192053Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_vqc_seed2_20260607T192053Z_summary.json",
        ],
        "metric": "eval_reward_mean",
    },
    {
            "label": "Distill VQC L3 raw",
        "params": 46,
        "method": "Fuzzy distill",
        "json": [
            "outputs/results/fuzzy_distill_dqn_vqc_seed0_20260607T190408Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_vqc_seed1_20260607T190849Z_summary.json",
            "outputs/results/fuzzy_distill_dqn_vqc_seed2_20260607T190826Z_summary.json",
        ],
        "metric": "eval_reward_mean",
    },
]


def _load_values(paths: list[str], metric: str) -> tuple[list[float], list[float]]:
    values = []
    times = []
    for path in paths:
        with (ROOT / path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for run in payload["runs"]:
            values.append(float(run[metric]))
            times.append(float(run.get("total_wall_time_sec", 0.0)))
    return values, times


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def main() -> None:
    rows = []
    for group in GROUPS:
        values, times = _load_values(group["json"], group["metric"])
        mean, std = _mean_std(values)
        time_mean, _ = _mean_std(times)
        rows.append(
            {
                "label": group["label"],
                "method": group["method"],
                "params": group["params"],
                "seeds": len(values),
                "eval_reward_mean": mean,
                "eval_reward_std": std,
                "time_sec_mean": time_mean,
            }
        )

    output_csv = ROOT / "outputs/tables/fuzzy_final_scenarios.csv"
    output_tex = ROOT / "outputs/tables/fuzzy_final_scenarios.tex"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\hline",
        "Scenario & Method & Params & Seeds & Eval reward & Time (s) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['method']} & {row['params']} & {row['seeds']} & "
            f"{_fmt(row['eval_reward_mean'], row['eval_reward_std'])} & "
            f"{row['time_sec_mean']:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    output_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"table_csv={output_csv}")
    print(f"table_tex={output_tex}")


if __name__ == "__main__":
    main()
