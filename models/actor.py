"""
MASOS - Actor Network.
obs(486) -> Linear(486,128) -> ReLU -> Linear(128,128) -> ReLU -> Linear(128,5) -> Softmax
Parameter sharing across all agents.
"""
import torch
import torch.nn as nn
from models.networks import MLP


class Actor(nn.Module):
    """
    Policy network shared across all agents.
    Outputs a categorical distribution over 5 actions.

    Args:
        obs_dim: Observation dimension (486)
        act_dim: Action dimension (5)
        hidden_dim: Hidden layer size (128)
    """

    def __init__(self, obs_dim: int = 486, act_dim: int = 5,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(
            input_dim=obs_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=act_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) or (obs_dim,)
        Returns:
            action_probs: (batch, act_dim) softmax probabilities
        """
        logits = self.net(obs)
        return torch.softmax(logits, dim=-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action from the policy.

        Args:
            obs: (obs_dim,) single observation
            deterministic: If True, return argmax action

        Returns:
            action: int, selected action
            log_prob: log probability of the selected action
            entropy: entropy of the distribution
        """
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor):
        """
        Evaluate log probabilities and entropy for given obs-action pairs.

        Args:
            obs: (batch, obs_dim)
            actions: (batch,) integer actions

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
