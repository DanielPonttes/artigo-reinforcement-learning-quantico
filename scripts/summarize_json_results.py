#!/usr/bin/env python3
"""Summarize run summary JSON files, including greedy evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument("--output-csv", default="outputs/tables/json_results_summary.csv")
    parser.add_argument("--output-tex", default="outputs/tables/json_results_summary.tex")
    return parser.parse_args()


def _agent_label(agent: str) -> str:
    return {
        "dqn_mlp": "DQN-MLP",
        "dqn_vqc": "DQN-VQC",
        "random_policy": "Random",
        "heuristic_policy": "Heuristic",
    }.get(agent, agent.replace("_", "\\_"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "--"
    return f"{value:.2f}"


def main() -> None:
    args = parse_args()
    runs = []
    for path in args.json:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        runs.extend(payload.get("runs", []))

    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(str(run["agent"]), []).append(run)

    rows = []
    for agent, agent_runs in sorted(grouped.items()):
        params = [float(run.get("trainable_parameters", 0)) for run in agent_runs]
        train_final = [
            float(run["final_20_mean_reward"])
            for run in agent_runs
            if "final_20_mean_reward" in run
        ]
        train_final += [
            float(run["final_10_mean_reward"])
            for run in agent_runs
            if "final_10_mean_reward" in run
        ]
        greedy = [
            float(run["greedy_eval_reward_mean"])
            for run in agent_runs
            if "greedy_eval_reward_mean" in run
        ]
        greedy += [
            float(run["eval_reward_mean"])
            for run in agent_runs
            if "eval_reward_mean" in run
        ]
        if not greedy and train_final:
            greedy = train_final
        best = [
            float(run["best_reward"])
            for run in agent_runs
            if "best_reward" in run
        ]
        time_sec = [float(run.get("total_wall_time_sec", 0.0)) for run in agent_runs]
        per_step = [
            float(run["wall_time_per_step_sec"])
            for run in agent_runs
            if "wall_time_per_step_sec" in run
        ]
        final_eps = [
            float(run["final_epsilon"])
            for run in agent_runs
            if "final_epsilon" in run
        ]
        train_mean, train_std = _mean_std(train_final)
        greedy_mean, greedy_std = _mean_std(greedy)
        best_mean, _ = _mean_std(best)
        time_mean, _ = _mean_std(time_sec)
        per_step_mean, _ = _mean_std(per_step)
        eps_mean, _ = _mean_std(final_eps)

        rows.append(
            {
                "agent": agent,
                "label": _agent_label(agent),
                "seeds": len(agent_runs),
                "trainable_parameters": int(round(np.mean(params))) if params else 0,
                "train_final_reward_mean": train_mean,
                "train_final_reward_std": train_std,
                "greedy_eval_reward_mean": greedy_mean,
                "greedy_eval_reward_std": greedy_std,
                "best_reward_mean": best_mean,
                "time_sec_mean": time_mean,
                "wall_time_per_step_sec_mean": per_step_mean,
                "final_epsilon_mean": eps_mean,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_tex = Path(args.output_tex)
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Agent & Params & Seeds & Train final & Greedy eval & Time (s) \\\\",
        "\\hline",
    ]
    for row in rows:
        train = f"{_fmt(row['train_final_reward_mean'])} $\\pm$ {_fmt(row['train_final_reward_std'])}"
        greedy = f"{_fmt(row['greedy_eval_reward_mean'])} $\\pm$ {_fmt(row['greedy_eval_reward_std'])}"
        lines.append(
            f"{row['label']} & {row['trainable_parameters']} & {row['seeds']} & "
            f"{train} & {greedy} & {_fmt(row['time_sec_mean'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    output_tex.write_text("\n".join(lines), encoding="utf-8")

    print(f"summary_csv={output_csv}")
    print(f"summary_tex={output_tex}")


if __name__ == "__main__":
    main()
