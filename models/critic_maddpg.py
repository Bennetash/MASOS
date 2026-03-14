"""
MASOS - MADDPG Critic.
Centralized critic: input = concat(all_obs, all_acts) = 8*486 + 8*5 = 3928-dim
Linear(3928,256) -> ReLU -> Linear(256,256) -> ReLU -> Linear(256,1)
Each agent has its own critic (no parameter sharing).
"""
import torch
import torch.nn as nn
from models.networks import MLP


class MADDPGCritic(nn.Module):
    """
    MADDPG centralized critic for a single agent.
    Takes all agents' observations and actions as input.

    Args:
        n_agents: Number of agents (8)
        obs_dim: Per-agent observation dimension (486)
        act_dim: Per-agent action dimension (5)
        hidden_dim: Hidden layer size (256)
    """

    def __init__(self, n_agents: int = 8, obs_dim: int = 486,
                 act_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        total_input = n_agents * (obs_dim + act_dim)  # 8 * 491 = 3928
        self.net = MLP(
            input_dim=total_input,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=1,
        )

    def forward(self, all_obs: torch.Tensor,
                all_actions_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_obs: (batch, n_agents, obs_dim) or (batch, n_agents * obs_dim)
            all_actions_onehot: (batch, n_agents, act_dim) or (batch, n_agents * act_dim)

        Returns:
            q_value: (batch, 1)
        """
        batch_size = all_obs.size(0)
        obs_flat = all_obs.reshape(batch_size, -1)
        act_flat = all_actions_onehot.reshape(batch_size, -1)
        x = torch.cat([obs_flat, act_flat], dim=-1)
        return self.net(x)
