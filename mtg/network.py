from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from .features import STATE_DIM, ACTION_DIM


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        if np.random.random() < epsilon:
            legal_indices = np.where(legal_mask)[0]
            return int(np.random.choice(legal_indices))

        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.forward(state_t).squeeze(0).numpy()

        q_values[~legal_mask] = float("-inf")
        return int(np.argmax(q_values))
