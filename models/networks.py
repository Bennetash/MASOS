"""
MASOS - MLP building block for neural network modules.
"""
import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Multi-Layer Perceptron building block.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function class (default: ReLU)
        output_activation: Optional output activation (default: None)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: type = nn.ReLU,
        output_activation: type = None,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
