import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Multi-layer perceptron used to approximate Q(s, a) for discrete Mario Kart actions.

    The network accepts a feature-based state vector and outputs one Q-value per action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """
        Initializes the Q-network architecture.

        Args:
            state_dim (int): Dimension of the input feature vector.
            action_dim (int): Number of discrete actions.
            hidden_dim (int): Number of hidden units per fully connected layer.

        Returns:
            None
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass through the Q-network.

        Args:
            state (torch.Tensor): Batch of input states with shape [batch_size, state_dim].

        Returns:
            torch.Tensor: Predicted Q-values with shape [batch_size, action_dim].
        """
        return self.model(state)