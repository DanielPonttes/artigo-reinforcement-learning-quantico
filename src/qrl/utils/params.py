"""Parameter-counting utilities."""

from __future__ import annotations

import torch.nn as nn


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def count_all_parameters(module: nn.Module) -> int:
    """Return the total number of parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in module.parameters())

