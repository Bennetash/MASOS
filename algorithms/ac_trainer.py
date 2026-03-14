"""
MASOS - Actor-Critic (AC) Trainer.
Follows Algorithm 1 from the paper.
Baseline: each agent uses local observations only.
Actor LR: 0.0005, Critic LR: 0.0005 (paper: AC uses 5e-4 for both), entropy_coef: 0.01
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from models.actor import Actor
from models.critic_ac import ACCritic
from algorithms.gae import compute_gae
from algorithms.rollout_buffer import RolloutBuffer
from configs.default import TrainingConfig


class ACTrainer:
    """
    Vanilla Actor-Critic trainer with GAE.

    Uses parameter sharing: one Actor and one Critic for all agents.

    Args:
        config: TrainingConfig instance
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self.n_agents = config.n_agents
        self.gamma = config.gamma
        self.lam = config.lam
        self.entropy_coef = config.entropy_coef

        # Shared networks
        self.actor = Actor(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.critic = ACCritic(
            obs_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.ac_actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.ac_critic_lr
        )

        # Buffer
        self.buffer = RolloutBuffer(n_agents=config.n_agents, device=self.device)

    def select_actions(self, obs_dict: Dict[int, np.ndarray],
                       alive_mask: Optional[Dict[int, bool]] = None) -> Dict:
        """
        Select actions for all agents using the shared actor.

        Args:
            obs_dict: {agent_id: obs_array}
            alive_mask: {agent_id: bool} optional, True=alive

        Returns:
            Dictionary with actions, log_probs, values for each agent
        """
        actions = {}
        log_probs = {}
        values = {}

        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            for i in range(self.n_agents):
                if alive_mask and not alive_mask.get(i, True):
                    actions[i] = 4  # dead agents stay
                    log_probs[i] = 0.0
                    values[i] = 0.0
                    continue

                obs_tensor = torch.tensor(
                    obs_dict[i], dtype=torch.float32, device=self.device
                )
                action, log_prob, _ = self.actor.get_action(obs_tensor)
                value = self.critic(obs_tensor.unsqueeze(0))

                actions[i] = action.item()
                log_probs[i] = log_prob.item()
                values[i] = value.squeeze().item()

        self.actor.train()
        self.critic.train()

        return {"actions": actions, "log_probs": log_probs, "values": values}

    def store_transition(
        self,
        obs_dict: Dict[int, np.ndarray],
        actions_dict: Dict[int, int],
        rewards_dict: Dict[int, float],
        values_dict: Dict[int, float],
        log_probs_dict: Dict[int, float],
        dones_dict: Dict,
    ):
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs_dict, actions_dict, rewards_dict,
                       values_dict, log_probs_dict, dones_dict)

    def update(self, next_obs_dict: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Perform multi-epoch mini-batch policy gradient update.

        1. Compute GAE advantages and returns ONCE (from rollout values).
        2. For n_update_epochs passes:
            - Shuffle data into mini-batches
            - For each mini-batch: compute actor + critic losses, update.
        3. Report averaged metrics across all mini-batch steps.

        Args:
            next_obs_dict: Final observations for computing bootstrap value

        Returns:
            Dictionary of loss metrics
        """
        data = self.buffer.get_tensors()
        T = data["global_dones"].size(0)
        n_update_epochs = self.config.n_update_epochs
        mini_batch_size = self.config.mini_batch_size

        # ============================================================
        # PHASE 1: Compute GAE advantages and returns ONCE per agent
        # ============================================================
        agent_advantages = {}
        agent_returns = {}
        for i in range(self.n_agents):
            obs_i = data["observations"][i]
            rewards_i = data["rewards"][i]
            old_values_i = data["values"][i]
            dones_i = data["dones"][i]

            with torch.no_grad():
                next_obs_tensor = torch.tensor(
                    next_obs_dict[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                next_value = self.critic(next_obs_tensor).squeeze()

            advantages, returns = compute_gae(
                rewards=rewards_i,
                values=old_values_i,
                next_value=next_value,
                dones=dones_i,
                gamma=self.gamma,
                lam=self.lam,
            )
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent_advantages[i] = advantages.detach()
            agent_returns[i] = returns.detach()

        # ============================================================
        # PHASE 2: Multi-epoch mini-batch updates
        # ============================================================
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch_idx in range(n_update_epochs):
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, mini_batch_size):
                end = min(start + mini_batch_size, T)
                mb_indices = indices[start:end]
                if mb_indices.size(0) < 2:
                    continue

                mb_actor_loss = 0.0
                mb_critic_loss = 0.0
                mb_entropy = 0.0

                for i in range(self.n_agents):
                    mb_obs_i = data["observations"][i][mb_indices]
                    mb_actions_i = data["actions"][i][mb_indices]
                    mb_advantages_i = agent_advantages[i][mb_indices]
                    mb_returns_i = agent_returns[i][mb_indices]

                    # Critic loss
                    new_values = self.critic(mb_obs_i).squeeze()
                    critic_loss = nn.functional.mse_loss(new_values, mb_returns_i)

                    # Actor loss
                    log_probs_new, entropy = self.actor.evaluate_actions(
                        mb_obs_i, mb_actions_i
                    )
                    actor_loss = -(log_probs_new * mb_advantages_i).mean()
                    actor_loss -= self.entropy_coef * entropy.mean()

                    mb_actor_loss += actor_loss
                    mb_critic_loss += critic_loss
                    mb_entropy += entropy.mean().item()

                avg_actor_loss = mb_actor_loss / self.n_agents
                avg_critic_loss = mb_critic_loss / self.n_agents

                # Update actor
                self.actor_optimizer.zero_grad()
                avg_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                avg_critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                total_actor_loss += avg_actor_loss.item()
                total_critic_loss += avg_critic_loss.item()
                total_entropy += mb_entropy / self.n_agents
                n_updates += 1

        # Clear buffer
        self.buffer.clear()

        n_updates = max(n_updates, 1)
        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def get_models(self) -> Dict[str, nn.Module]:
        """Return models for checkpointing."""
        return {"actor": self.actor, "critic": self.critic}

    def load_models(self, state_dicts: Dict[str, dict]):
        """Load model state dicts."""
        self.actor.load_state_dict(state_dicts["actor"])
        self.critic.load_state_dict(state_dicts["critic"])
