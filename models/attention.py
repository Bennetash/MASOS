"""
MASOS - Multi-Head Attention with Pre-LayerNorm.
Implements the attention mechanism from the paper (Section III.B).
Key innovation: LayerNorm BEFORE attention computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional Pre-LayerNorm.

    Args:
        d_model: Model dimension (128)
        n_heads: Number of attention heads (4)
        d_k: Key/Query dimension per head (32)
        use_layer_norm: Whether to apply LayerNorm before attention
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 d_k: int = 32, use_layer_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.use_layer_norm = use_layer_norm

        # Pre-LayerNorm (paper's innovation)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_k, bias=False)

        # Output projection
        self.W_o = nn.Linear(n_heads * d_k, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: (batch, n_query, d_model)  - for agent i
            key:   (batch, n_key, d_model)    - all agents
            value: (batch, n_key, d_model)    - all agents

        Args (added):
            mask: (batch, n_key) optional alive mask. True=alive, False=dead.
                  Dead agents get -inf attention logit -> zero weight.

        Returns:
            output: (batch, n_query, d_model)
        """
        batch_size = query.size(0)

        # Pre-LayerNorm
        if self.use_layer_norm:
            query = self.layer_norm(query)
            key = self.layer_norm(key)
            value = self.layer_norm(value)

        # Project to Q, K, V: (batch, seq, n_heads * d_k)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape to (batch, n_heads, seq, d_k)
        n_q = query.size(1)
        n_k = key.size(1)
        Q = Q.view(batch_size, n_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, n_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, n_k, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, n_heads, n_q, d_k) x (batch, n_heads, d_k, n_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Mask dead agents with -inf so they get zero attention weight
        if mask is not None:
            # mask: (batch, n_key) where True=alive, False=dead
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_key)
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, n_q, n_k)

        # Apply attention to values
        # (batch, n_heads, n_q, n_k) x (batch, n_heads, n_k, d_k)
        context = torch.matmul(attn_weights, V)  # (batch, n_heads, n_q, d_k)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, n_q, self.n_heads * self.d_k
        )
        output = self.W_o(context)  # (batch, n_q, d_model)

        return output


class StackedMultiHeadAttention(nn.Module):
    """
    Stack of N MultiHeadAttention layers with Pre-LN residual connections.
    Paper: "stack multiple" attention layers with LayerNorm BEFORE attention.

    Pre-LN architecture (per paper Section III.B):
        output = output + attn(LN(output), LN(key), LN(value))

    Each layer has its own LayerNorm for the query stream (Pre-LN on query)
    and a shared-per-layer LayerNorm for key/value normalization.

    Args:
        n_layers: Number of stacked attention layers (default 2)
        d_model: Model dimension (128)
        n_heads: Number of attention heads (4)
        d_k: Key/Query dimension per head (32)
    """

    def __init__(self, n_layers: int = 2, d_model: int = 128,
                 n_heads: int = 4, d_k: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                use_layer_norm=False,  # LN handled here in the stack, not inside attn
            )
            for _ in range(n_layers)
        ])
        # Pre-LayerNorm: one LN per layer for the query stream
        self.query_layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        # Pre-LayerNorm: one LN per layer for the key/value stream
        self.kv_layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pre-LN forward pass:
            For each layer:
                normed_query = LN_query(output)
                normed_kv    = LN_kv(key)  (key and value share same source)
                output       = output + attn(normed_query, normed_kv, normed_kv)

        Args:
            query: (batch, n_query, d_model)
            key:   (batch, n_key, d_model)
            value: (batch, n_key, d_model)
            mask:  (batch, n_key) optional alive mask (True=alive)

        Returns:
            output: (batch, n_query, d_model)
        """
        output = query
        for attn_layer, q_ln, kv_ln in zip(self.layers, self.query_layer_norms,
                                            self.kv_layer_norms):
            # Pre-LN: normalize BEFORE attention (paper Eq. 5-6)
            normed_query = q_ln(output)
            normed_kv = kv_ln(key)  # key == value (same source encodings)
            # Cross-attention with normalized inputs
            attn_out = attn_layer(normed_query, normed_kv, normed_kv, mask=mask)
            # Residual connection (NO post-LN)
            output = output + attn_out
        return output
