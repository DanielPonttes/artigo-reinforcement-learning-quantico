#!/usr/bin/env python3
"""Small smoke tests for core components."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrl.agents.replay_buffer import ReplayBuffer
from qrl.models.factory import create_q_network
from qrl.utils.params import count_trainable_parameters
from qrl.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vqc", action="store_true")
    return parser.parse_args()


def test_replay_buffer() -> None:
    buffer = ReplayBuffer(capacity=10, seed=0)
    for idx in range(5):
        state = np.full(4, idx, dtype=np.float32)
        buffer.push(state, idx % 2, 1.0, state + 1.0, False)
    batch = buffer.sample(3)
    assert batch.states.shape == (3, 4)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.next_states.shape == (3, 4)
    assert batch.dones.shape == (3,)


def test_mlp_forward() -> None:
    model = create_q_network("dqn_mlp", 4, 2, {"hidden_sizes": [16]})
    x = torch.zeros(3, 4)
    y = model(x)
    assert y.shape == (3, 2)
    assert count_trainable_parameters(model) == 114


def test_vqc_forward() -> None:
    model = create_q_network(
        "dqn_vqc",
        4,
        2,
        {
            "n_qubits": 4,
            "n_layers": 1,
            "backend": "default.qubit",
            "data_reuploading": True,
        },
    )
    x = torch.zeros(2, 4)
    y = model(x)
    assert y.shape == (2, 2)
    assert count_trainable_parameters(model) == 22


def main() -> None:
    args = parse_args()
    set_global_seed(0)
    test_replay_buffer()
    test_mlp_forward()
    if not args.skip_vqc:
        test_vqc_forward()
    print("smoke_test=ok")


if __name__ == "__main__":
    main()

