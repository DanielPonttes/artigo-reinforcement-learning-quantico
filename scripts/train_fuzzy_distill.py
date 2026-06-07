#!/usr/bin/env python3
"""Distill a fuzzy/heuristic CartPole teacher into a small MLP or VQC policy."""

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
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.envs.preprocessing import preprocess_observation
from qrl.models.factory import create_q_network
from qrl.utils.params import count_trainable_parameters
from qrl.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--agent", choices=["dqn_mlp", "dqn_vqc"], required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _teacher_action(observation: np.ndarray) -> int:
    x, velocity, theta, angular_velocity = observation
    score = theta + 0.25 * angular_velocity + 0.01 * x + 0.01 * velocity
    return 1 if score > 0 else 0


def _sample_cartpole_states(samples: int, rng: np.random.Generator) -> np.ndarray:
    low = np.asarray([-2.4, -3.0, -0.2095, -3.0], dtype=np.float32)
    high = np.asarray([2.4, 3.0, 0.2095, 3.0], dtype=np.float32)
    return rng.uniform(low=low, high=high, size=(samples, 4)).astype(np.float32)


def _evaluate(model: torch.nn.Module, config: dict, seed: int, episodes: int) -> dict:
    env_id = str(config["env_id"])
    feature_mode = str(config.get("feature_mode", "raw_normalized"))
    env = gym.make(env_id)
    rewards = []
    steps_list = []
    model.eval()
    for episode in range(episodes):
        raw_state, _ = env.reset(seed=200000 + seed * 1000 + episode)
        state = preprocess_observation(raw_state, env_id, feature_mode=feature_mode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            with torch.no_grad():
                q_values = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())
            raw_state, reward, terminated, truncated, _ = env.step(action)
            state = preprocess_observation(raw_state, env_id, feature_mode=feature_mode)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
        rewards.append(total_reward)
        steps_list.append(steps)
    env.close()
    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "eval_steps_mean": float(np.mean(steps_list)),
    }


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["agent"] = args.agent

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = gym.make(str(config["env_id"]))
    input_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    env.close()

    model = create_q_network(
        agent_name=args.agent,
        input_dim=input_dim,
        n_actions=n_actions,
        model_config=dict(config.get("model", {})),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    raw_states = _sample_cartpole_states(args.samples, rng)
    feature_mode = str(config.get("feature_mode", "raw_normalized"))
    features = np.stack(
        [
            preprocess_observation(raw_state, str(config["env_id"]), feature_mode=feature_mode)
            for raw_state in raw_states
        ]
    )
    labels = np.asarray([_teacher_action(raw_state) for raw_state in raw_states], dtype=np.int64)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.int64)

    started = time.perf_counter()
    losses = []
    for epoch in range(args.epochs):
        permutation = torch.randperm(args.samples)
        for start in range(0, args.samples, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits = model(x[indices])
            loss = F.cross_entropy(logits, y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            with torch.no_grad():
                accuracy = (torch.argmax(model(x), dim=1) == y).float().mean().item()
            print(
                f"progress epoch={epoch + 1}/{args.epochs} loss={np.mean(losses[-10:]):.4f} acc={accuracy:.3f}",
                flush=True,
            )

    with torch.no_grad():
        train_accuracy = (torch.argmax(model(x), dim=1) == y).float().mean().item()
    eval_summary = _evaluate(model, config, seed=args.seed, episodes=args.eval_episodes)

    timestamp = _timestamp()
    output_root = Path(args.output_dir)
    results_dir = output_root / "results"
    checkpoints_dir = output_root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"fuzzy_distill_{args.agent}_seed{args.seed}_{timestamp}"
    checkpoint = checkpoints_dir / f"{run_id}.pt"
    torch.save(
        {
            "run_id": run_id,
            "seed": args.seed,
            "config": config,
            "model_state_dict": model.state_dict(),
            "trainable_parameters": count_trainable_parameters(model),
        },
        checkpoint,
    )

    summary = {
        "run_id": run_id,
        "experiment": config.get("experiment", ""),
        "agent": args.agent,
        "env": config["env_id"],
        "feature_mode": config.get("feature_mode", "raw_normalized"),
        "model_config": config.get("model", {}),
        "seed": args.seed,
        "teacher": "fuzzy_heuristic",
        "samples": args.samples,
        "epochs": args.epochs,
        "train_accuracy": float(train_accuracy),
        "trainable_parameters": count_trainable_parameters(model),
        "total_wall_time_sec": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        **eval_summary,
    }

    summary_json = results_dir / f"{run_id}_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({"runs": [summary]}, handle, indent=2)

    csv_path = results_dir / f"{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"summary_json={summary_json}")
    print(f"checkpoint={checkpoint}")
    print(f"eval_reward_mean={summary['eval_reward_mean']}")


if __name__ == "__main__":
    main()
