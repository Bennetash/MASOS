"""
MASOS - Actor-Critic Baseline Critic.
obs(486) -> Linear(486,128) -> ReLU -> Linear(128,128) -> ReLU -> Linear(128,1)
Estimates V(s) from local observation.
"""
import torch
import torch.nn as nn
from models.networks import MLP


class ACCritic(nn.Module):
    """
    Vanilla Actor-Critic value function.
    Shared across all agents (parameter sharing).

    Args:
        obs_dim: Observation dimension (486)
        hidden_dim: Hidden layer size (128)
    """

    def __init__(self, obs_dim: int = 486, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(
            input_dim=obs_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=1,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) or (obs_dim,)
        Returns:
            value: (batch, 1) state value estimate
        """
        return self.net(obs)
