"""
MASOS - On-policy rollout buffer for AC and MAAC trainers.
Stores transitions from environment rollouts for policy gradient updates.
"""
import torch
import numpy as np
from typing import Dict, List, Optional


class RolloutBuffer:
    """
    On-policy rollout buffer for multi-agent actor-critic methods.

    Stores observations, actions, rewards, values, log_probs, and dones
    for each agent over one episode/rollout.
    """

    def __init__(self, n_agents: int, device: str = "cpu"):
        self.n_agents = n_agents
        self.device = device
        self.clear()

    def clear(self):
        """Reset all buffers."""
        self.observations = {i: [] for i in range(self.n_agents)}
        self.actions = {i: [] for i in range(self.n_agents)}
        self.rewards = {i: [] for i in range(self.n_agents)}
        self.values = {i: [] for i in range(self.n_agents)}
        self.log_probs = {i: [] for i in range(self.n_agents)}
        self.dones = {i: [] for i in range(self.n_agents)}
        self.global_dones = []
        # For MAAC: store all observations at each step
        self.all_observations = []  # list of (n_agents, obs_dim) arrays
        self.all_actions = []       # list of (n_agents,) arrays
        self._size = 0

    def add(
        self,
        obs_dict: Dict[int, np.ndarray],
        actions_dict: Dict[int, int],
        rewards_dict: Dict[int, float],
        values_dict: Dict[int, float],
        log_probs_dict: Dict[int, float],
        dones_dict: Dict,
    ):
        """
        Add a single timestep transition for all agents.

        Args:
            obs_dict: {agent_id: obs_array}
            actions_dict: {agent_id: action_int}
            rewards_dict: {agent_id: reward_float}
            values_dict: {agent_id: value_float}
            log_probs_dict: {agent_id: log_prob_float}
            dones_dict: {"agent_0": bool, ..., "__all__": bool}
        """
        for i in range(self.n_agents):
            self.observations[i].append(obs_dict[i])
            self.actions[i].append(actions_dict[i])
            self.rewards[i].append(rewards_dict[i])
            self.values[i].append(values_dict[i])
            self.log_probs[i].append(log_probs_dict[i])
            self.dones[i].append(float(dones_dict.get(f"agent_{i}", False)))

        self.global_dones.append(float(dones_dict.get("__all__", False)))

        # Store joint observations and actions
        all_obs = np.stack([obs_dict[i] for i in range(self.n_agents)])
        all_acts = np.array([actions_dict[i] for i in range(self.n_agents)])
        self.all_observations.append(all_obs)
        self.all_actions.append(all_acts)
        self._size += 1

    def get_tensors(self) -> Dict:
        """
        Convert all stored data to PyTorch tensors.

        Returns:
            Dictionary containing tensors for each agent and shared data.
        """
        device = self.device
        data = {
            "observations": {},
            "actions": {},
            "rewards": {},
            "values": {},
            "log_probs": {},
            "dones": {},  # per-agent dones
            "global_dones": torch.tensor(self.global_dones, dtype=torch.float32, device=device),
            "all_observations": torch.tensor(
                np.array(self.all_observations), dtype=torch.float32, device=device
            ),  # (T, n_agents, obs_dim)
            "all_actions": torch.tensor(
                np.array(self.all_actions), dtype=torch.long, device=device
            ),  # (T, n_agents)
        }

        for i in range(self.n_agents):
            data["observations"][i] = torch.tensor(
                np.array(self.observations[i]), dtype=torch.float32, device=device
            )
            data["actions"][i] = torch.tensor(
                self.actions[i], dtype=torch.long, device=device
            )
            data["rewards"][i] = torch.tensor(
                self.rewards[i], dtype=torch.float32, device=device
            )
            data["values"][i] = torch.tensor(
                self.values[i], dtype=torch.float32, device=device
            )
            data["log_probs"][i] = torch.tensor(
                self.log_probs[i], dtype=torch.float32, device=device
            )
            data["dones"][i] = torch.tensor(
                self.dones[i], dtype=torch.float32, device=device
            )

        return data

    @property
    def size(self) -> int:
        return self._size
