from __future__ import annotations

import torch
from torch import nn


class SquashedMLPPolicy(nn.Module):
    """Readable policy network with explicit torque squashing."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int], max_action: float) -> None:
        super().__init__()
        hidden_one, hidden_two = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_one),
            nn.Tanh(),
            nn.Linear(hidden_one, hidden_two),
            nn.Tanh(),
            nn.Linear(hidden_two, 1),
        )
        self.max_action = float(max_action)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(features))


class PositiveMLPPolicy(nn.Module):
    """Policy with outputs constrained to the positive assist range."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int], max_action: float) -> None:
        super().__init__()
        hidden_one, hidden_two = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_one),
            nn.Tanh(),
            nn.Linear(hidden_one, hidden_two),
            nn.Tanh(),
            nn.Linear(hidden_two, 1),
        )
        self.max_action = float(max_action)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.sigmoid(self.network(features))
