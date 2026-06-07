"""Q-network factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from qrl.models.mlp import MLPQNetwork


def create_q_network(
    agent_name: str,
    input_dim: int,
    n_actions: int,
    model_config: dict[str, Any],
) -> nn.Module:
    dueling = bool(model_config.get("dueling", False))

    if agent_name == "dqn_mlp":
        return MLPQNetwork(
            input_dim=input_dim,
            n_actions=n_actions,
            hidden_sizes=list(model_config.get("hidden_sizes", [16])),
            dueling=dueling,
        )

    if agent_name == "dqn_vqc":
        from qrl.models.vqc import VQCQNetwork

        return VQCQNetwork(
            input_dim=input_dim,
            n_actions=n_actions,
            n_qubits=int(model_config.get("n_qubits", input_dim)),
            n_layers=int(model_config.get("n_layers", 1)),
            backend=str(model_config.get("backend", "default.qubit")),
            data_reuploading=bool(model_config.get("data_reuploading", True)),
            dueling=dueling,
        )

    raise ValueError(f"Unknown agent: {agent_name}")

