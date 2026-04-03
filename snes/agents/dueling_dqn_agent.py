"""
Dueling DQN agent factory.

Builds a Tianshou DQNPolicy configured as:
  - Double DQN        (is_double=True)
  - Dueling network   (via DuelingQNetwork architecture)
  - N-step returns    (estimation_step=n_step)
  - Epsilon-greedy exploration with linear decay

The caller (train.py) is responsible for the replay buffer and collector.
"""

import torch
from tianshou.policy import DQNPolicy
from .q_network import DuelingQNetwork


def make_dueling_agent(cfg, n_actions: int, n_stack: int, device: str, action_space):
    """
    Args:
        cfg:          dueling section of config (cfg.dueling)
        n_actions:    number of discrete actions
        n_stack:      number of stacked frames
        device:       'cpu' or 'cuda'
        action_space: gymnasium action space (for tianshou policy)

    Returns:
        policy:       DQNPolicy ready for training
    """
    network = DuelingQNetwork(
        n_actions  = n_actions,
        n_stack    = n_stack,
        hidden_dim = cfg["hidden_dim"],
    ).to(device)

    # Separate target network (tianshou creates this internally via deepcopy)
    optim = torch.optim.Adam(network.parameters(), lr=cfg["lr"])

    policy = DQNPolicy(
        model               = network,
        optim               = optim,
        discount_factor     = cfg["gamma"],
        estimation_step     = cfg["n_step"],
        target_update_freq  = cfg["target_update_freq"],
        is_double           = True,   # Double DQN: target network selects actions
        action_space        = action_space,
    )

    return policy


def make_train_fn(policy, cfg, total_epochs: int, step_per_epoch: int):
    """Returns a train_fn callback for offpolicy_trainer that linearly decays epsilon."""
    eps_start      = cfg["eps_train_start"]
    eps_end        = cfg["eps_train_end"]
    eps_decay_steps = cfg["eps_decay_steps"]

    def train_fn(epoch, env_step):
        eps = max(eps_end, eps_start - env_step * (eps_start - eps_end) / eps_decay_steps)
        policy.set_eps(eps)

    return train_fn
