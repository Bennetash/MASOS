"""
MASOS - MAAC (Multi-Actor-Attention-Critic) Trainer.
Same as AC but critic uses multi-head attention across agents.
CTDE: Centralized Training (critic sees all agents), Decentralized Execution (actor sees own obs).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from models.actor import Actor
from models.critic_maac import MAACCritic
from algorithms.gae import compute_gae
from algorithms.rollout_buffer import RolloutBuffer
from configs.default import TrainingConfig


class MAACTrainer:
    """
    Multi-Actor-Attention-Critic trainer with GAE.

    Centralized critic with attention, shared actor for decentralized execution.

    Args:
        config: TrainingConfig instance
        use_layer_norm: Whether to use LayerNorm in attention (for ablation)
    """

    def __init__(self, config: TrainingConfig, use_layer_norm: bool = True):
        self.config = config
        self.device = config.device
        self.n_agents = config.n_agents
        self.gamma = config.gamma
        self.lam = config.lam
        self.entropy_coef = config.entropy_coef

        # Shared actor (decentralized execution)
        self.actor = Actor(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Centralized critic with attention
        self.critic = MAACCritic(
            n_agents=config.n_agents,
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_attention_layers=config.n_attention_layers,
            use_layer_norm=use_layer_norm,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        # Buffer
        self.buffer = RolloutBuffer(n_agents=config.n_agents, device=self.device)

    def select_actions(self, obs_dict: Dict[int, np.ndarray],
                       alive_mask: Optional[Dict[int, bool]] = None) -> Dict:
        """
        Select actions for all agents using the shared actor.
        Also compute values using the centralized critic.

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
            # Get actions from actor (decentralized)
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
                actions[i] = action.item()
                log_probs[i] = log_prob.item()

            # Get values from centralized critic
            all_obs = torch.tensor(
                np.stack([obs_dict[i] for i in range(self.n_agents)]),
                dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, n_agents, obs_dim)

            for i in range(self.n_agents):
                if alive_mask and not alive_mask.get(i, True):
                    continue  # already set to 0.0 above
                value = self.critic.get_value(all_obs, agent_idx=i)
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

        # Joint data for centralized critic
        all_obs = data["all_observations"]   # (T, n_agents, obs_dim)
        all_actions = data["all_actions"]     # (T, n_agents)
        per_agent_dones = data["dones"]       # {agent_id: (T,)} per-agent

        # Next obs for bootstrapping
        next_all_obs = torch.tensor(
            np.stack([next_obs_dict[i] for i in range(self.n_agents)]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, n_agents, obs_dim)

        # ============================================================
        # PHASE 1: Compute GAE advantages and returns ONCE per agent
        # ============================================================
        agent_advantages = {}
        agent_returns = {}
        for i in range(self.n_agents):
            rewards_i = data["rewards"][i]
            old_values_i = data["values"][i]

            with torch.no_grad():
                next_value = self.critic.get_value(next_all_obs, agent_idx=i).squeeze()

            advantages, returns = compute_gae(
                rewards=rewards_i,
                values=old_values_i,
                next_value=next_value,
                dones=per_agent_dones[i],
                gamma=self.gamma,
                lam=self.lam,
            )
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent_advantages[i] = advantages.detach()
            agent_returns[i] = returns.detach()

        # ============================================================
        # PHASE 2: Multi-epoch mini-batch updates
        # ============================================================
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch_idx in range(n_update_epochs):
            # Generate random permutation for this epoch
            indices = torch.randperm(T, device=self.device)

            # Split into mini-batches
            for start in range(0, T, mini_batch_size):
                end = min(start + mini_batch_size, T)
                mb_indices = indices[start:end]
                mb_size = mb_indices.size(0)
                if mb_size < 2:
                    continue  # skip tiny batches

                # Slice joint data for this mini-batch
                mb_all_obs = all_obs[mb_indices]         # (mb, n_agents, obs_dim)
                mb_all_actions = all_actions[mb_indices]  # (mb, n_agents)

                mb_actor_loss = 0.0
                mb_critic_loss = 0.0
                mb_q_loss = 0.0
                mb_v_loss = 0.0
                mb_entropy = 0.0

                for i in range(self.n_agents):
                    mb_obs_i = data["observations"][i][mb_indices]
                    mb_actions_i = data["actions"][i][mb_indices]
                    mb_advantages_i = agent_advantages[i][mb_indices]
                    mb_returns_i = agent_returns[i][mb_indices]

                    # Critic Q-head loss
                    q_values = self.critic(mb_all_obs, mb_all_actions, agent_idx=i)
                    q_values = q_values.squeeze(-1)
                    q_loss = nn.functional.mse_loss(q_values, mb_returns_i)

                    # Critic V-head loss
                    v_values = self.critic.get_value(mb_all_obs, agent_idx=i)
                    v_values = v_values.squeeze(-1)
                    v_loss = nn.functional.mse_loss(v_values, mb_returns_i)

                    critic_loss = q_loss + v_loss

                    # Actor loss
                    log_probs_new, entropy = self.actor.evaluate_actions(
                        mb_obs_i, mb_actions_i
                    )
                    actor_loss = -(log_probs_new * mb_advantages_i).mean()
                    actor_loss -= self.entropy_coef * entropy.mean()

                    mb_actor_loss += actor_loss
                    mb_critic_loss += critic_loss
                    mb_q_loss += q_loss.item()
                    mb_v_loss += v_loss.item()
                    mb_entropy += entropy.mean().item()

                # Average over agents
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
                total_q_loss += mb_q_loss / self.n_agents
                total_v_loss += mb_v_loss / self.n_agents
                total_entropy += mb_entropy / self.n_agents
                n_updates += 1

        # Clear buffer
        self.buffer.clear()

        # Average over all mini-batch updates
        n_updates = max(n_updates, 1)
        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "q_loss": total_q_loss / n_updates,
            "v_loss": total_v_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def get_models(self) -> Dict[str, nn.Module]:
        """Return models for checkpointing."""
        return {"actor": self.actor, "critic": self.critic}

    def load_models(self, state_dicts: Dict[str, dict]):
        """Load model state dicts."""
        self.actor.load_state_dict(state_dicts["actor"])
        self.critic.load_state_dict(state_dicts["critic"])
