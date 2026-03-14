"""
MASOS - Generalized Advantage Estimation (GAE).
Implements Equations (3) and (4) from the paper.

Equation (3): delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
Equation (4): A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

Default: gamma=0.99, lambda=0.97
"""
import torch
import numpy as np
from typing import List


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.97,
) -> tuple:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: (T,) rewards at each timestep
        values: (T,) value estimates V(s_t)
        next_value: (1,) value estimate V(s_{T+1})
        dones: (T,) done flags (1.0 if episode ended)
        gamma: Discount factor
        lam: GAE lambda

    Returns:
        advantages: (T,) GAE advantages
        returns: (T,) discounted returns (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)

    last_gae = 0.0
    last_value = next_value.squeeze()

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        # Equation (3): TD residual
        delta = rewards[t] + gamma * last_value * mask - values[t]
        # Equation (4): GAE accumulation
        advantages[t] = delta + gamma * lam * mask * last_gae
        last_gae = advantages[t]
        last_value = values[t]

    returns = advantages + values
    return advantages, returns


def compute_gae_multi_agent(
    rewards_dict: dict,
    values_dict: dict,
    next_values_dict: dict,
    dones_dict: dict,
    gamma: float = 0.99,
    lam: float = 0.97,
) -> tuple:
    """
    Compute GAE for multiple agents with per-agent done masks.

    Args:
        rewards_dict: {agent_id: (T,) tensor}
        values_dict: {agent_id: (T,) tensor}
        next_values_dict: {agent_id: scalar tensor}
        dones_dict: {agent_id: (T,) tensor} per-agent done flags
        gamma: Discount factor
        lam: GAE lambda

    Returns:
        advantages_dict: {agent_id: (T,) tensor}
        returns_dict: {agent_id: (T,) tensor}
    """
    advantages_dict = {}
    returns_dict = {}

    for agent_id in rewards_dict:
        adv, ret = compute_gae(
            rewards=rewards_dict[agent_id],
            values=values_dict[agent_id],
            next_value=next_values_dict[agent_id],
            dones=dones_dict[agent_id],
            gamma=gamma,
            lam=lam,
        )
        advantages_dict[agent_id] = adv
        returns_dict[agent_id] = ret

    return advantages_dict, returns_dict
