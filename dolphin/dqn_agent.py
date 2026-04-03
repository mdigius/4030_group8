from typing import Any, Dict, Tuple
import numpy as np
import torch

from dolphin.q_network import QNetwork


class DQNAgent:
    """
    Baseline Deep Q-Network (DQN) agent skeleton for the Mario Kart project.

    This class defines the modular structure for the baseline value-based RL agent
    proposed in Phase 1. In Phase 2, the purpose of this file is to establish the
    architecture, method responsibilities, and interfaces needed for Phase 3.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the DQN agent skeleton.

        Args:
            state_dim (int): Dimension of the feature-based state vector.
            action_dim (int): Number of discrete actions available to the agent.
            learning_rate (float): Optimizer learning rate.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial epsilon value for exploration.
            epsilon_min (float): Minimum epsilon value.
            epsilon_decay (float): Epsilon decay factor or rate.
            device (str): Compute device identifier (e.g., 'cpu', 'cuda', 'mps').

        Returns:
            None
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device

        self.q_network = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        self.target_network = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.replay_buffer = []
        self.loss_history = []

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current environment state vector.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: Selected discrete action index.
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores one transition in replay memory.

        Args:
            state (np.ndarray): Current state.
            action (int): Executed action.
            reward (float): Observed reward.
            next_state (np.ndarray): Next observed state.
            done (bool): Whether the episode terminated.

        Returns:
            None
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, batch_size: int) -> Dict[str, float]:
        """
        Performs one DQN learning/update step from replayed experience.

        Args:
            batch_size (int): Number of transitions to sample from replay memory.

        Returns:
            Dict[str, float]: Dictionary of training statistics such as loss.
        """
        # Phase 2 skeleton placeholder
        return {"loss": 0.0}

    def update_epsilon(self) -> None:
        """
        Updates the exploration rate according to the configured schedule.

        Args:
            None

        Returns:
            None
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        """
        Copies weights from the online Q-network to the target network.

        Args:
            None

        Returns:
            None
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath: str) -> None:
        """
        Saves the online Q-network weights to disk.

        Args:
            filepath (str): Destination path for the saved model file.

        Returns:
            None
        """
        torch.save(self.q_network.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Loads previously saved Q-network weights from disk.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            None
        """
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())