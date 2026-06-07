"""DQN training loop used by the experiment scripts."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from qrl.agents.replay_buffer import ReplayBuffer
from qrl.envs.preprocessing import preprocess_observation
from qrl.models.factory import create_q_network
from qrl.utils.params import count_trainable_parameters
from qrl.utils.seeding import set_global_seed


@dataclass(frozen=True)
class ExperimentArtifacts:
    results_csv: Path
    summary_json: Path
    config_json: Path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_device(agent_name: str, requested_device: str, backend: str = "default.qubit") -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if agent_name == "dqn_vqc" and "gpu" not in backend:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _linear_epsilon(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> float:
    if epsilon_decay_steps <= 0:
        return epsilon_end
    progress = min(1.0, step / float(epsilon_decay_steps))
    return epsilon_start + progress * (epsilon_end - epsilon_start)


def _cartpole_fuzzy_action(raw_observation: np.ndarray) -> int:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = raw_observation
    score = pole_angle + 0.25 * pole_angular_velocity + 0.01 * cart_position + 0.01 * cart_velocity
    return 1 if score > 0 else 0


def _select_action(
    policy_net: torch.nn.Module,
    state: np.ndarray,
    raw_state: np.ndarray,
    env_id: str,
    epsilon: float,
    n_actions: int,
    device: torch.device,
    rng: np.random.Generator,
    exploration_policy: str,
) -> int:
    if rng.random() < epsilon:
        if exploration_policy == "fuzzy_guided" and env_id == "CartPole-v1":
            return _cartpole_fuzzy_action(raw_state)
        return int(rng.integers(n_actions))

    with torch.no_grad():
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(tensor_state)
        return int(torch.argmax(q_values, dim=1).item())


def _evaluate_greedy_policy(
    policy_net: torch.nn.Module,
    env_id: str,
    feature_mode: str,
    seed: int,
    episodes: int,
    device: torch.device,
) -> dict[str, float] | None:
    if episodes <= 0:
        return None

    env = gym.make(env_id)
    rewards: list[float] = []
    steps_list: list[int] = []
    policy_net.eval()

    try:
        for episode in range(episodes):
            raw_state, _ = env.reset(seed=100000 + seed * 1000 + episode)
            state = preprocess_observation(raw_state, env_id, feature_mode=feature_mode)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action = int(torch.argmax(policy_net(state_tensor), dim=1).item())
                raw_state, reward, terminated, truncated, _ = env.step(action)
                state = preprocess_observation(raw_state, env_id, feature_mode=feature_mode)
                done = bool(terminated or truncated)
                total_reward += float(reward)
                steps += 1

            rewards.append(total_reward)
            steps_list.append(steps)
    finally:
        env.close()
        policy_net.train()

    return {
        "greedy_eval_episodes": float(episodes),
        "greedy_eval_reward_mean": float(np.mean(rewards)),
        "greedy_eval_reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "greedy_eval_steps_mean": float(np.mean(steps_list)),
    }


def _optimize_model(
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    gradient_clip_norm: float | None,
    double_dqn: bool = False,
) -> float | None:
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = target_net(next_states).max(dim=1).values
        targets = rewards + gamma * (1.0 - dones) * next_q_values

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip_norm)
    optimizer.step()

    return float(loss.item())


def train_one_seed(
    config: dict[str, Any],
    seed: int,
    run_id: str,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env_id = str(config["env_id"])
    agent_name = str(config["agent"])
    model_config = dict(config.get("model", {}))
    episodes = int(config.get("episodes", 500))
    gamma = float(config.get("gamma", 0.99))
    learning_rate = float(config.get("learning_rate", 1e-3))
    batch_size = int(config.get("batch_size", 64))
    buffer_size = int(config.get("buffer_size", 50000))
    target_update_interval = int(config.get("target_update_interval", 500))
    epsilon_start = float(config.get("epsilon_start", 1.0))
    epsilon_end = float(config.get("epsilon_end", 0.05))
    epsilon_decay_steps = int(config.get("epsilon_decay_steps", 10000))
    learning_starts = int(config.get("learning_starts", batch_size))
    max_steps_per_episode = int(config.get("max_steps_per_episode", 100000))
    double_dqn = bool(config.get("double_dqn", False))
    gradient_clip_norm = config.get("gradient_clip_norm", 10.0)
    requested_device = str(config.get("device", "auto"))
    log_interval_episodes = int(config.get("log_interval_episodes", 10))
    evaluation_episodes = int(config.get("evaluation_episodes", 0))
    feature_mode = str(config.get("feature_mode", "raw_normalized"))
    exploration_policy = str(config.get("exploration_policy", "epsilon_random"))

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    device = _resolve_device(agent_name, requested_device, backend=str(model_config.get("backend", "default.qubit")))

    env = gym.make(env_id)
    env.action_space.seed(seed)

    obs_space = env.observation_space
    if not hasattr(obs_space, "shape") or obs_space.shape is None:
        raise ValueError(f"Only vector observation spaces are supported, got {obs_space}")

    input_dim = int(np.prod(obs_space.shape))
    n_actions = int(env.action_space.n)
    policy_net = create_q_network(agent_name, input_dim, n_actions, model_config).to(device)
    target_net = create_q_network(agent_name, input_dim, n_actions, model_config).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    global_step = 0
    started_at = time.perf_counter()

    for episode in range(episodes):
        raw_state, _ = env.reset(seed=seed + episode)
        state = preprocess_observation(raw_state, env_id, feature_mode=feature_mode)
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done and episode_steps < max_steps_per_episode:
            epsilon = _linear_epsilon(
                global_step,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay_steps=epsilon_decay_steps,
            )
            action = _select_action(
                policy_net,
                state=state,
                raw_state=raw_state,
                env_id=env_id,
                epsilon=epsilon,
                n_actions=n_actions,
                device=device,
                rng=rng,
                exploration_policy=exploration_policy,
            )
            raw_next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_state = preprocess_observation(raw_next_state, env_id, feature_mode=feature_mode)

            replay_buffer.push(state, action, reward, next_state, done)
            raw_state = raw_next_state
            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            global_step += 1

            if len(replay_buffer) >= learning_starts:
                loss = _optimize_model(
                    policy_net=policy_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                    gradient_clip_norm=float(gradient_clip_norm)
                    if gradient_clip_norm is not None
                    else None,
                    double_dqn=double_dqn,
                )
                if loss is not None:
                    losses.append(loss)

            if target_update_interval > 0 and global_step % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rows.append(
            {
                "run_id": run_id,
                "agent": agent_name,
                "env": env_id,
                "seed": seed,
                "episode": episode,
                "reward": episode_reward,
                "steps": episode_steps,
                "wall_time_sec": time.perf_counter() - started_at,
                "epsilon": _linear_epsilon(
                    global_step,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay_steps=epsilon_decay_steps,
                ),
                "global_step": global_step,
            }
        )

        if log_interval_episodes > 0 and (
            (episode + 1) % log_interval_episodes == 0 or episode + 1 == episodes
        ):
            elapsed = time.perf_counter() - started_at
            print(
                f"progress run_id={run_id} episode={episode + 1}/{episodes} "
                f"steps={global_step} reward={episode_reward:.1f} elapsed_sec={elapsed:.2f}",
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "agent": agent_name,
        "env": env_id,
        "seed": seed,
        "episodes": episodes,
        "global_steps": global_step,
        "total_wall_time_sec": time.perf_counter() - started_at,
        "trainable_parameters": count_trainable_parameters(policy_net),
        "device": str(device),
        "final_epsilon": _linear_epsilon(
            global_step,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        ),
        "wall_time_per_step_sec": (time.perf_counter() - started_at) / max(global_step, 1),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "final_10_mean_reward": float(np.mean([row["reward"] for row in rows[-10:]]))
        if rows
        else None,
    }

    greedy_eval = _evaluate_greedy_policy(
        policy_net=policy_net,
        env_id=env_id,
        feature_mode=feature_mode,
        seed=seed,
        episodes=evaluation_episodes,
        device=device,
    )
    if greedy_eval is not None:
        summary.update(greedy_eval)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "run_id": run_id,
                "seed": seed,
                "config": config,
                "model_state_dict": policy_net.state_dict(),
                "trainable_parameters": count_trainable_parameters(policy_net),
            },
            checkpoint_path,
        )
        summary["checkpoint"] = str(checkpoint_path)

    env.close()
    return rows, summary


def run_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> ExperimentArtifacts:
    experiment_name = str(config.get("experiment", config.get("agent", "experiment")))
    timestamp = _utc_timestamp()
    output_root = Path(output_dir or config.get("outputs_dir", "outputs"))
    results_dir = output_root / "results"
    checkpoints_dir = output_root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in config.get("seeds", [0])]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    base_name = f"{experiment_name}_{timestamp}"
    results_csv = results_dir / f"{base_name}.csv"
    summary_json = results_dir / f"{base_name}_summary.json"
    config_json = results_dir / f"{base_name}_config.json"

    with config_json.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for seed in seeds:
        run_id = f"{experiment_name}_seed{seed}_{timestamp}"
        checkpoint_path = checkpoints_dir / f"{run_id}.pt"
        rows, summary = train_one_seed(
            config,
            seed=seed,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
        )
        all_rows.extend(rows)
        summaries.append(summary)

        if all_rows:
            write_header = not results_csv.exists()
            with results_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)

        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump({"runs": summaries}, handle, indent=2)

    return ExperimentArtifacts(
        results_csv=results_csv,
        summary_json=summary_json,
        config_json=config_json,
    )
