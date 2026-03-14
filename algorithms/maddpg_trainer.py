"""
MASOS - MADDPG Trainer.
Off-policy multi-agent DDPG with centralized critics.
Uses replay buffer. LR=1e-4, batch=512, tau=0.01.
Each agent has its own critic (no parameter sharing for critics).
Actors share parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import copy

from models.actor import Actor
from models.critic_maddpg import MADDPGCritic
from algorithms.replay_buffer import ReplayBuffer
from configs.default import TrainingConfig


class MADDPGTrainer:
    """
    MADDPG trainer.

    Each agent has a separate centralized critic.
    Actor uses parameter sharing (shared across agents).
    Uses Gumbel-Softmax for differentiable discrete actions.

    Args:
        config: TrainingConfig instance
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self.n_agents = config.n_agents
        self.tau = config.maddpg_tau
        self.batch_size = config.maddpg_batch_size

        # Shared actor
        self.actor = Actor(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Per-agent critics
        self.critics = nn.ModuleList([
            MADDPGCritic(
                n_agents=config.n_agents,
                obs_dim=config.obs_dim,
                act_dim=config.act_dim,
                hidden_dim=config.maddpg_hidden_dim,
            ).to(self.device)
            for _ in range(config.n_agents)
        ])

        # Target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critics = nn.ModuleList([
            copy.deepcopy(c) for c in self.critics
        ])

        # Freeze targets
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for critic in self.target_critics:
            for p in critic.parameters():
                p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.maddpg_lr
        )
        self.critic_optimizers = [
            torch.optim.Adam(self.critics[i].parameters(), lr=config.maddpg_lr)
            for i in range(config.n_agents)
        ]

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.maddpg_buffer_size,
            n_agents=config.n_agents,
            obs_dim=config.obs_dim,
            device=self.device,
        )

        # Exploration noise (epsilon-greedy)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def select_actions(
        self, obs_dict: Dict[int, np.ndarray], explore: bool = True
    ) -> Dict:
        """
        Select actions for all agents.
        Uses epsilon-greedy exploration during training.

        Args:
            obs_dict: {agent_id: obs_array}
            explore: Whether to add exploration noise

        Returns:
            Dictionary with actions for each agent
        """
        actions = {}

        self.actor.eval()
        with torch.no_grad():
            for i in range(self.n_agents):
                obs_tensor = torch.tensor(
                    obs_dict[i], dtype=torch.float32, device=self.device
                )
                action, _, _ = self.actor.get_action(obs_tensor, deterministic=not explore)

                if explore and np.random.random() < self.epsilon:
                    actions[i] = np.random.randint(0, self.config.act_dim)
                else:
                    actions[i] = action.item()

        self.actor.train()

        # Decay epsilon
        if explore:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"actions": actions}

    def store_transition(
        self,
        obs_dict: Dict[int, np.ndarray],
        actions_dict: Dict[int, int],
        rewards_dict: Dict[int, float],
        next_obs_dict: Dict[int, np.ndarray],
        dones_dict: Dict,
    ):
        """Store a transition in the replay buffer."""
        self.buffer.add(obs_dict, actions_dict, rewards_dict, next_obs_dict, dones_dict)

    def update(self) -> Dict[str, float]:
        """
        Perform one update step using a sampled batch from replay buffer.

        Returns:
            Dictionary of loss metrics (or empty if buffer not ready)
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]           # (B, n_agents, obs_dim)
        actions = batch["actions"]   # (B, n_agents)
        rewards = batch["rewards"]   # (B, n_agents)
        next_obs = batch["next_obs"] # (B, n_agents, obs_dim)
        dones = batch["dones"]       # (B, n_agents)

        B = obs.size(0)
        act_dim = self.config.act_dim

        # One-hot encode actions
        actions_onehot = torch.zeros(
            B, self.n_agents, act_dim, device=self.device
        )
        actions_onehot.scatter_(2, actions.unsqueeze(-1), 1.0)

        # Compute target actions (one-hot from target actor)
        with torch.no_grad():
            target_actions_onehot = torch.zeros(
                B, self.n_agents, act_dim, device=self.device
            )
            for i in range(self.n_agents):
                target_probs = self.target_actor(next_obs[:, i, :])
                target_acts = target_probs.argmax(dim=-1)
                target_actions_onehot[:, i].scatter_(
                    1, target_acts.unsqueeze(-1), 1.0
                )

        # Update each critic
        total_critic_loss = 0.0
        for i in range(self.n_agents):
            # Target Q
            with torch.no_grad():
                target_q = self.target_critics[i](next_obs, target_actions_onehot)
                target_q = rewards[:, i:i+1] + (1.0 - dones[:, i:i+1]) * self.config.gamma * target_q

            # Current Q
            current_q = self.critics[i](obs, actions_onehot)
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=0.5)
            self.critic_optimizers[i].step()

            total_critic_loss += critic_loss.item()

        # Update actor using policy gradient through critic 0 (or average)
        # Use Gumbel-Softmax for differentiable discrete actions
        actor_actions_onehot = torch.zeros(
            B, self.n_agents, act_dim, device=self.device
        )
        for i in range(self.n_agents):
            logits = self.actor.net(obs[:, i, :])
            gumbel_actions = F.gumbel_softmax(logits, tau=1.0, hard=True)
            actor_actions_onehot[:, i] = gumbel_actions

        # Actor loss: maximize average Q across all critics
        total_actor_loss = 0.0
        for i in range(self.n_agents):
            actor_loss = -self.critics[i](obs, actor_actions_onehot).mean()
            total_actor_loss += actor_loss

        avg_actor_loss = total_actor_loss / self.n_agents

        self.actor_optimizer.zero_grad()
        avg_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update()

        return {
            "actor_loss": avg_actor_loss.item(),
            "critic_loss": total_critic_loss / self.n_agents,
            "epsilon": self.epsilon,
        }

    def _soft_update(self):
        """Polyak averaging update for target networks."""
        tau = self.tau
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

        for i in range(self.n_agents):
            for target_param, param in zip(
                self.target_critics[i].parameters(),
                self.critics[i].parameters(),
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

    def get_models(self) -> Dict[str, nn.Module]:
        """Return models for checkpointing."""
        models = {"actor": self.actor}
        for i in range(self.n_agents):
            models[f"critic_{i}"] = self.critics[i]
        return models

    def load_models(self, state_dicts: Dict[str, dict]):
        """Load model state dicts."""
        self.actor.load_state_dict(state_dicts["actor"])
        self.target_actor.load_state_dict(state_dicts["actor"])
        for i in range(self.n_agents):
            key = f"critic_{i}"
            if key in state_dicts:
                self.critics[i].load_state_dict(state_dicts[key])
                self.target_critics[i].load_state_dict(state_dicts[key])
