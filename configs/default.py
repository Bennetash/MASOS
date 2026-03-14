"""
MASOS - Default configuration and task definitions.
All hyperparameters match the paper: Pei & Luo, IEEE Systems Journal 2026.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TrainingConfig:
    # ---- Environment ----
    grid_size: int = 40
    n_agents: int = 8
    n_targets: int = 50
    n_obstacles: int = 3
    obs_radius: int = 5          # f-norm radius -> 11x11 FOV
    target_mode: str = "random"  # "static" or "random"
    max_steps: int = 500
    cooperation_distance: int = 3  # Manhattan distance for coop reward

    # ---- Rewards (Table II) ----
    reward_find_target: float = 50.0
    reward_collide_agent: float = -12.0
    reward_collide_obstacle: float = -12.0
    reward_stay: float = 0.0
    reward_move: float = -0.05
    reward_cooperation: float = 3.0
    reward_boundary_death: float = -12.0  # penalty for dying at boundary

    # ---- Observation ----
    # 4 channels * 11*11 + 2 (own position normalized)
    obs_dim: int = 486
    act_dim: int = 5  # up, down, left, right, stay

    # ---- Training (Algorithm 1) ----
    n_epochs: int = 6000
    actor_lr: float = 1e-4       # 0.0001
    critic_lr: float = 5e-4      # 0.0005
    gamma: float = 0.99
    lam: float = 0.97            # GAE lambda
    entropy_coef: float = 0.01   # beta > 0 in Algorithm 1
    n_update_epochs: int = 4     # number of passes over rollout data per update
    mini_batch_size: int = 128   # mini-batch size for each gradient step

    # ---- Network Architecture ----
    hidden_dim: int = 128
    n_heads: int = 4
    d_k: int = 32                # hidden_dim // n_heads
    n_attention_layers: int = 2  # Paper: "stack multiple" attention layers

    # ---- MADDPG-specific ----
    maddpg_lr: float = 1e-4
    maddpg_batch_size: int = 512
    maddpg_buffer_size: int = 1_000_000
    maddpg_tau: float = 0.01
    maddpg_hidden_dim: int = 256

    # ---- AC-specific (paper: AC uses 5e-4 for both) ----
    ac_actor_lr: float = 5e-4
    ac_critic_lr: float = 5e-4

    # ---- System ----
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 50
    save_interval: int = 200
    eval_episodes: int = 20

    def update(self, overrides: Dict[str, Any]):
        """Update config with a dictionary of overrides."""
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
        # Recompute derived values
        self.d_k = self.hidden_dim // self.n_heads
        return self


# 6 Task configurations from the paper (Section II.C)
TASK_CONFIGS = {
    1: {"grid_size": 40, "n_targets": 50,  "target_mode": "random"},
    2: {"grid_size": 40, "n_targets": 50,  "target_mode": "static"},
    3: {"grid_size": 80, "n_targets": 50,  "target_mode": "random"},
    4: {"grid_size": 80, "n_targets": 50,  "target_mode": "static"},
    5: {"grid_size": 40, "n_targets": 100, "target_mode": "random"},
    6: {"grid_size": 40, "n_targets": 100, "target_mode": "static"},
}
