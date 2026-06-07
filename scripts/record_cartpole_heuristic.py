#!/usr/bin/env python3
"""Render a simple CartPole heuristic for presentation-only visuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/videos/cartpole_heuristic_demo.gif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    observation, _ = env.reset(seed=args.seed)
    frames = []
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < args.max_steps:
        frames.append(env.render())
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        score = pole_angle + 0.25 * pole_angular_velocity + 0.01 * cart_position + 0.01 * cart_velocity
        action = 1 if score > 0 else 0
        observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
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

