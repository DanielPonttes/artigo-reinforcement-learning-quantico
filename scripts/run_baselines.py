#!/usr/bin/env python3
"""Run non-learning CartPole baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _heuristic_action(observation: np.ndarray) -> int:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
    score = pole_angle + 0.25 * pole_angular_velocity + 0.01 * cart_position + 0.01 * cart_velocity
    return 1 if score > 0 else 0


def _run_policy(env_id: str, policy_name: str, seed: int, episodes: int) -> tuple[list[dict], dict]:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    rows = []
    started = time.perf_counter()

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if policy_name == "random_policy":
                action = int(rng.integers(env.action_space.n))
            elif policy_name == "heuristic_policy":
                action = _heuristic_action(np.asarray(observation, dtype=np.float32))
            else:
                raise ValueError(f"Unknown baseline: {policy_name}")

            observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1

        rows.append(
            {
                "run_id": f"{policy_name}_seed{seed}",
                "agent": policy_name,
                "env": env_id,
                "seed": seed,
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "wall_time_sec": time.perf_counter() - started,
                "epsilon": "",
                "global_step": int(sum(row["steps"] for row in rows) + steps),
            }
        )

    env.close()
    rewards = [row["reward"] for row in rows]
    summary = {
        "agent": policy_name,
        "env": env_id,
        "seed": seed,
        "episodes": episodes,
        "trainable_parameters": 0,
        "final_20_mean_reward": float(np.mean(rewards[-20:])),
        "best_reward": float(np.max(rewards)),
        "total_wall_time_sec": time.perf_counter() - started,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    timestamp = _timestamp()
    output_root = Path(args.output_dir)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for policy_name in ["random_policy", "heuristic_policy"]:
        all_rows = []
        summaries = []
        for seed in args.seeds:
            rows, summary = _run_policy(args.env_id, policy_name, seed, args.episodes)
            run_id = f"{policy_name}_seed{seed}_{timestamp}"
            for row in rows:
                row["run_id"] = run_id
            summary["run_id"] = run_id
            all_rows.extend(rows)
            summaries.append(summary)

        results_csv = results_dir / f"{policy_name}_{timestamp}.csv"
        summary_json = results_dir / f"{policy_name}_{timestamp}_summary.json"
        with results_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump({"runs": summaries}, handle, indent=2)
        print(f"{policy_name}_csv={results_csv}")
        print(f"{policy_name}_summary={summary_json}")


if __name__ == "__main__":
    main()

