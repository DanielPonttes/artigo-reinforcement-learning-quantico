#!/usr/bin/env python3
"""Render a trained checkpoint as a GIF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.envs.preprocessing import preprocess_observation
from qrl.models.factory import create_q_network
from qrl.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env_id = str(config["env_id"])
    agent_name = str(config["agent"])
    model_config = dict(config.get("model", {}))
    feature_mode = str(config.get("feature_mode", "raw_normalized"))

    set_global_seed(args.seed)
    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    raw_state, _ = env.reset(seed=args.seed)

    input_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    model = create_q_network(agent_name, input_dim, n_actions, model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frames = []
    total_reward = 0.0
    steps = 0
    done = False
    state = preprocess_observation(raw_state, env_id, feature_mode=feature_mode)

    while not done and steps < args.max_steps:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action = int(torch.argmax(model(state_tensor), dim=1).item())
        raw_next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        state = preprocess_observation(raw_next_state, env_id, feature_mode=feature_mode)
        total_reward += float(reward)
        steps += 1

    if frames:
        frames.append(env.render())
    env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=args.fps)
    print(f"video={output_path}")
    print(f"steps={steps}")
    print(f"reward={total_reward}")


if __name__ == "__main__":
    main()
