"""Classical MLP Q-network baseline with optional dueling architecture."""

from __future__ import annotations

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    """Feed-forward Q-network with optional dueling head."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
        dueling: bool = False,
    ):
        super().__init__()
        self.dueling = bool(dueling)

        trunk_layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for hidden_size in hidden_sizes:
            trunk_layers.append(nn.Linear(current_dim, int(hidden_size)))
            trunk_layers.append(nn.ReLU())
            current_dim = int(hidden_size)

        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()

        if self.dueling:
            self.value_head = nn.Linear(current_dim, 1)
            self.advantage_head = nn.Linear(current_dim, int(n_actions))
        else:
            self.head = nn.Linear(current_dim, int(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        features = self.trunk(x.float())

        if self.dueling:
            value = self.value_head(features)
            advantages = self.advantage_head(features)
            return value + advantages - advantages.mean(dim=1, keepdim=True)

        return self.head(features)

