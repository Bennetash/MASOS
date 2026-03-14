"""
MASOS - MAAC Critic (Main Contribution).
Implements Equation (7): Q_i(o,a) = f_i(l_i(g_i(l_i(o_i, a_i))), x_i)

Architecture per Paper:
    1. l_i: LayerNorm on raw input (obs_i, act_i_onehot) -- 491 dim
    2. g_i: Single linear layer encoder + ReLU -> 128 dim
    3. l_i: LayerNorm on encoder output (pre-attention)
    4. Stacked Multi-Head Attention (2 layers, 4 heads, d_k=32) with residuals
    5. f_i: Output MLP concat(own_encoded, attn_output) -> Q value
    6. Separate centralized V(s) head with its own attention (no actions)
"""
import torch
import torch.nn as nn
from models.networks import MLP
from models.attention import StackedMultiHeadAttention


class MAACCritic(nn.Module):
    """
    Multi-Actor-Attention-Critic.

    Each agent i computes Q_i by attending to other agents' encoded
    observation-action pairs. Uses Pre-LayerNorm attention with stacked layers.

    Args:
        n_agents: Number of agents (8)
        obs_dim: Observation dimension (486)
        act_dim: Action dimension (5)
        hidden_dim: Hidden dimension (128)
        n_heads: Number of attention heads (4)
        n_attention_layers: Number of stacked attention layers (2)
        use_layer_norm: Whether to use LayerNorm (for ablation study)
    """

    def __init__(self, n_agents: int = 8, obs_dim: int = 486,
                 act_dim: int = 5, hidden_dim: int = 128,
                 n_heads: int = 4, n_attention_layers: int = 2,
                 use_layer_norm: bool = True):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # ===================== Q-HEAD PATH =====================
        # Eq(7): Q_i = f_i(l_i(g_i(l_i(o_i, a_i))), x_i)

        # l_i on raw input: LayerNorm BEFORE encoder (paper Eq 7, first l_i)
        self.input_layer_norm = (
            nn.LayerNorm(obs_dim + act_dim) if use_layer_norm else nn.Identity()
        )

        # g_i: SINGLE linear layer encoder (paper specifies single layer)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
        )

        # NOTE: Pre-attention LayerNorm (second l_i in Eq 7) is now handled
        # INSIDE StackedMultiHeadAttention as Pre-LN at each layer.
        # This avoids double-normalization and ensures every stacked layer
        # gets proper Pre-LN, not just the first one.

        # Stacked Multi-Head Attention with internal Pre-LN (paper: "stack multiple" layers)
        self.attention = StackedMultiHeadAttention(
            n_layers=n_attention_layers,
            d_model=hidden_dim,
            n_heads=n_heads,
            d_k=hidden_dim // n_heads,
        )

        # f_i: Output MLP -> Q value
        # Input: concat of own encoded + attention output = 2 * hidden_dim
        self.q_head = MLP(
            input_dim=2 * hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=1,
        )

        # ===================== V-HEAD PATH (Centralized) =====================
        # V(s) uses attention over all agents' OBSERVATIONS (no actions)

        self.v_input_layer_norm = (
            nn.LayerNorm(obs_dim) if use_layer_norm else nn.Identity()
        )
        self.v_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        # NOTE: V-head pre-attention LayerNorm is handled INSIDE the stack

        # V-head attention with internal Pre-LN (separate from Q-head attention)
        self.v_attention = StackedMultiHeadAttention(
            n_layers=n_attention_layers,
            d_model=hidden_dim,
            n_heads=n_heads,
            d_k=hidden_dim // n_heads,
        )

        # V-head output: concat(own_v_encoded, v_attn_output) = 2*hidden_dim
        self.v_head = MLP(
            input_dim=2 * hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=1,
        )

    def forward(self, all_obs: torch.Tensor,
                all_actions: torch.Tensor,
                agent_idx: int,
                alive_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Q value for agent_idx given all observations and actions.

        Args:
            all_obs: (batch, n_agents, obs_dim)
            all_actions: (batch, n_agents) integer actions
            agent_idx: which agent's Q value to compute
            alive_mask: (batch, n_agents) optional, True=alive

        Returns:
            q_value: (batch, 1)
        """
        batch_size = all_obs.size(0)

        # One-hot encode actions
        actions_onehot = torch.zeros(
            batch_size, self.n_agents, self.act_dim,
            device=all_obs.device, dtype=all_obs.dtype
        )
        actions_onehot.scatter_(2, all_actions.unsqueeze(-1).long(), 1.0)

        # Eq(7) step 1: l_i on raw input
        encoder_input = torch.cat([all_obs, actions_onehot], dim=-1)
        encoder_input = self.input_layer_norm(encoder_input)

        # Eq(7) step 2: g_i single-layer encoder
        encoded = self.encoder(encoder_input)

        # Eq(7) steps 3-4: Pre-LN + Stacked Multi-Head Attention
        # Pre-attention LayerNorm is now handled INSIDE the stack at each layer
        query = encoded[:, agent_idx:agent_idx+1, :]
        key = encoded
        value = encoded

        attn_out = self.attention(query, key, value, mask=alive_mask)
        attn_out = attn_out.squeeze(1)

        # Own encoding
        own_encoded = encoded[:, agent_idx, :]

        # Eq(7) step 5: f_i output
        combined = torch.cat([own_encoded, attn_out], dim=-1)
        q_value = self.q_head(combined)

        return q_value

    def get_value(self, all_obs: torch.Tensor,
                  agent_idx: int,
                  alive_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute V(s) for agent_idx using CENTRALIZED attention over observations.

        Args:
            all_obs: (batch, n_agents, obs_dim)
            agent_idx: which agent
            alive_mask: (batch, n_agents) optional, True=alive

        Returns:
            value: (batch, 1)
        """
        # Encode observations only (no actions)
        normed_obs = self.v_input_layer_norm(all_obs)
        v_encoded = self.v_encoder(normed_obs)

        # Pre-LN is handled INSIDE the stack at each layer
        query = v_encoded[:, agent_idx:agent_idx+1, :]
        key = v_encoded
        value_kv = v_encoded

        # Attention with internal Pre-LN
        attn_out = self.v_attention(query, key, value_kv, mask=alive_mask)
        attn_out = attn_out.squeeze(1)

        # Combine own encoding with attention output
        own_v_encoded = v_encoded[:, agent_idx, :]
        combined = torch.cat([own_v_encoded, attn_out], dim=-1)
        v_value = self.v_head(combined)

        return v_value
