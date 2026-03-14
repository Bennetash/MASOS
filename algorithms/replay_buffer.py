"""
MASOS - Off-policy replay buffer for MADDPG trainer.
Uniform random sampling from stored transitions.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional


class ReplayBuffer:
    """
    Experience replay buffer for multi-agent off-policy learning (MADDPG).

    Stores (obs, actions, rewards, next_obs, dones) tuples for all agents.

    Args:
        capacity: Maximum buffer size
        n_agents: Number of agents
        obs_dim: Per-agent observation dimension
        device: PyTorch device
    """

    def __init__(self, capacity: int, n_agents: int, obs_dim: int,
                 device: str = "cpu"):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.device = device

        # Pre-allocate numpy arrays
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs_dict: Dict[int, np.ndarray],
        actions_dict: Dict[int, int],
        rewards_dict: Dict[int, float],
        next_obs_dict: Dict[int, np.ndarray],
        dones_dict: Dict,
    ):
        """Add a single transition for all agents."""
        idx = self.ptr

        for i in range(self.n_agents):
            self.obs[idx, i] = obs_dict[i]
            self.actions[idx, i] = actions_dict[i]
            self.rewards[idx, i] = rewards_dict[i]
            self.next_obs[idx, i] = next_obs_dict[i]

        for i in range(self.n_agents):
            self.dones[idx, i] = float(dones_dict.get(f"agent_{i}", False))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of tensors: obs, actions, rewards, next_obs, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        device = self.device

        return {
            "obs": torch.tensor(
                self.obs[indices], dtype=torch.float32, device=device
            ),  # (batch, n_agents, obs_dim)
            "actions": torch.tensor(
                self.actions[indices], dtype=torch.long, device=device
            ),  # (batch, n_agents)
            "rewards": torch.tensor(
                self.rewards[indices], dtype=torch.float32, device=device
            ),  # (batch, n_agents)
            "next_obs": torch.tensor(
                self.next_obs[indices], dtype=torch.float32, device=device
            ),  # (batch, n_agents, obs_dim)
            "dones": torch.tensor(
                self.dones[indices], dtype=torch.float32, device=device
            ),  # (batch, n_agents)
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size
