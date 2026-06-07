#!/usr/bin/env python3
"""Unit tests for training utilities, preprocessing, and action selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.envs.preprocessing import preprocess_observation
from qrl.training.dqn import _linear_epsilon, _cartpole_fuzzy_action, _select_action
from qrl.models.factory import create_q_network
from qrl.agents.replay_buffer import ReplayBuffer
from qrl.utils.params import count_trainable_parameters, count_all_parameters
from qrl.utils.seeding import set_global_seed


def test_preprocessing_raw_normalized() -> None:
    obs = np.array([0.0, 0.0, 0.1, 1.0], dtype=np.float32)
    result = preprocess_observation(obs, "CartPole-v1", "raw_normalized")
    assert result.shape == (4,)
    assert np.all(result >= -np.pi)
    assert np.all(result <= np.pi)


def test_preprocessing_fuzzy_membership_4() -> None:
    obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = preprocess_observation(obs, "CartPole-v1", "fuzzy_membership_4")
    assert result.shape == (4,)
    assert np.all(result >= -np.pi)
    assert np.all(result <= np.pi)


def test_preprocessing_fuzzy_compact() -> None:
    obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = preprocess_observation(obs, "CartPole-v1", "fuzzy_compact")
    assert result.shape == (4,)
    assert np.all(result >= -np.pi)
    assert np.all(result <= np.pi)


def test_preprocessing_clipping() -> None:
    obs = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    result = preprocess_observation(obs, "CartPole-v1", "raw_normalized")
    assert result.shape == (4,)
    assert np.all(np.abs(result) <= np.pi)


def test_preprocessing_passthrough() -> None:
    obs = np.array([0.5, -0.3, 0.1, 0.2], dtype=np.float32)
    result = preprocess_observation(obs, "UnknownEnv-v0", "raw_normalized")
    np.testing.assert_array_almost_equal(result, obs)


def test_linear_epsilon_start() -> None:
    eps = _linear_epsilon(0, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=400)
    assert abs(eps - 1.0) < 1e-6


def test_linear_epsilon_end() -> None:
    eps = _linear_epsilon(1000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=400)
    assert abs(eps - 0.05) < 1e-6


def test_linear_epsilon_midpoint() -> None:
    eps = _linear_epsilon(200, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=400)
    expected = 1.0 + 0.5 * (0.05 - 1.0)
    assert abs(eps - expected) < 1e-4


def test_linear_epsilon_zero_decay() -> None:
    eps = _linear_epsilon(100, epsilon_start=0.5, epsilon_end=0.1, epsilon_decay_steps=0)
    assert abs(eps - 0.1) < 1e-6


def test_fuzzy_action_left() -> None:
    obs = np.array([0.0, 0.0, -0.2, -1.0], dtype=np.float32)
    action = _cartpole_fuzzy_action(obs)
    assert action == 0


def test_fuzzy_action_right() -> None:
    obs = np.array([0.0, 0.0, 0.2, 1.0], dtype=np.float32)
    action = _cartpole_fuzzy_action(obs)
    assert action == 1


def test_fuzzy_action_boundary() -> None:
    obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action = _cartpole_fuzzy_action(obs)
    assert action == 0


def test_replay_buffer_overflow() -> None:
    buffer = ReplayBuffer(capacity=3, seed=0)
    for i in range(10):
        state = np.full(4, float(i), dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)
    assert len(buffer) == 3


def test_replay_buffer_circular() -> None:
    buffer = ReplayBuffer(capacity=3, seed=0)
    buffer.push(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0, 1.0, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), False)
    buffer.push(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32), 1, 2.0, np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32), True)
    buffer.push(np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32), 0, 3.0, np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32), False)
    buffer.push(np.array([6.0, 6.0, 6.0, 6.0], dtype=np.float32), 1, 4.0, np.array([7.0, 7.0, 7.0, 7.0], dtype=np.float32), False)
    assert len(buffer) == 3


def test_seed_reproducibility() -> None:
    set_global_seed(42)
    rng1 = np.random.default_rng(42)
    val1 = rng1.random()

    set_global_seed(42)
    rng2 = np.random.default_rng(42)
    val2 = rng2.random()

    assert abs(val1 - val2) < 1e-10


def test_select_action_greedy() -> None:
    set_global_seed(0)
    model = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [8]})
    model.eval()
    state = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
    raw_state = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    action = _select_action(
        policy_net=model,
        state=state,
        raw_state=raw_state,
        env_id="CartPole-v1",
        epsilon=0.0,
        n_actions=2,
        device=device,
        rng=rng,
        exploration_policy="epsilon_random",
    )
    assert action in (0, 1)


def test_select_action_random() -> None:
    set_global_seed(0)
    model = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [8]})
    model.eval()
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(99)
    device = torch.device("cpu")

    actions = set()
    for _ in range(50):
        action = _select_action(
            policy_net=model,
            state=state,
            raw_state=raw_state,
            env_id="CartPole-v1",
            epsilon=1.0,
            n_actions=2,
            device=device,
            rng=rng,
            exploration_policy="epsilon_random",
        )
        actions.add(action)
    assert actions == {0, 1}


def test_select_action_fuzzy_guided() -> None:
    set_global_seed(0)
    model = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [8]})
    model.eval()
    device = torch.device("cpu")
    rng = np.random.default_rng(0)

    pole_right_state = np.array([0.0, 0.0, 0.2, 1.0], dtype=np.float32)
    pole_left_state = np.array([0.0, 0.0, -0.2, -1.0], dtype=np.float32)

    action_right = _select_action(
        policy_net=model,
        state=pole_right_state,
        raw_state=pole_right_state,
        env_id="CartPole-v1",
        epsilon=1.0,
        n_actions=2,
        device=device,
        rng=rng,
        exploration_policy="fuzzy_guided",
    )
    action_left = _select_action(
        policy_net=model,
        state=pole_left_state,
        raw_state=pole_left_state,
        env_id="CartPole-v1",
        epsilon=1.0,
        n_actions=2,
        device=device,
        rng=rng,
        exploration_policy="fuzzy_guided",
    )
    assert action_right == 1
    assert action_left == 0


def test_mlp_dueling_forward() -> None:
    model = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [8], "dueling": True})
    x = torch.zeros(3, 4)
    y = model(x)
    assert y.shape == (3, 2)


def test_mlp_dueling_param_count() -> None:
    model_base = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [8], "dueling": False})
    model_dueling = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [8], "dueling": True})
    base_params = count_trainable_parameters(model_base)
    dueling_params = count_trainable_parameters(model_dueling)
    assert base_params > 0
    assert dueling_params > base_params


def test_vqc_dueling_forward() -> None:
    model = create_q_network(
        "dqn_vqc", 4, 2,
        {"n_qubits": 4, "n_layers": 1, "backend": "default.qubit", "dueling": True},
    )
    x = torch.zeros(2, 4)
    y = model(x)
    assert y.shape == (2, 2)


def test_vqc_dueling_param_count() -> None:
    model_base = create_q_network(
        "dqn_vqc", 4, 2,
        {"n_qubits": 4, "n_layers": 1, "backend": "default.qubit", "dueling": False},
    )
    model_dueling = create_q_network(
        "dqn_vqc", 4, 2,
        {"n_qubits": 4, "n_layers": 1, "backend": "default.qubit", "dueling": True},
    )
    base_params = count_trainable_parameters(model_base)
    dueling_params = count_trainable_parameters(model_dueling)
    assert base_params > 0
    assert dueling_params > base_params


def test_vqc_batched_forward() -> None:
    model = create_q_network(
        "dqn_vqc", 4, 2,
        {"n_qubits": 4, "n_layers": 1, "backend": "default.qubit"},
    )
    x = torch.randn(16, 4)
    y = model(x)
    assert y.shape == (16, 2)


def test_parameter_counts_consistency() -> None:
    for agent_name in ["dqn_mlp", "dqn_vqc"]:
        model_config = (
            {"hidden_sizes": [8]}
            if agent_name == "dqn_mlp"
            else {"n_qubits": 4, "n_layers": 1, "backend": "default.qubit"}
        )
        model = create_q_network(agent_name, 4, 2, model_config)
        trainable = count_trainable_parameters(model)
        total = count_all_parameters(model)
        assert trainable == total
        assert trainable > 0


def main() -> None:
    set_global_seed(0)
    tests = [
        test_preprocessing_raw_normalized,
        test_preprocessing_fuzzy_membership_4,
        test_preprocessing_fuzzy_compact,
        test_preprocessing_clipping,
        test_preprocessing_passthrough,
        test_linear_epsilon_start,
        test_linear_epsilon_end,
        test_linear_epsilon_midpoint,
        test_linear_epsilon_zero_decay,
        test_fuzzy_action_left,
        test_fuzzy_action_right,
        test_fuzzy_action_boundary,
        test_replay_buffer_overflow,
        test_replay_buffer_circular,
        test_seed_reproducibility,
        test_select_action_greedy,
        test_select_action_random,
        test_select_action_fuzzy_guided,
        test_mlp_dueling_forward,
        test_mlp_dueling_param_count,
        test_vqc_dueling_forward,
        test_vqc_dueling_param_count,
        test_vqc_batched_forward,
        test_parameter_counts_consistency,
    ]

    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1

    if failed:
        print(f"\ntest_training=fail ({failed}/{len(tests)} failed)")
        sys.exit(1)
    print(f"\ntest_training=ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
