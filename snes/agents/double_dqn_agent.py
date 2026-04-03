"""
Double DQN agent factory.

Standard CNN Q-network trained with Double DQN:
  - Online network selects the greedy action
  - Target network evaluates that action's Q-value
  - Eliminates overestimation bias present in vanilla DQN

No dueling streams, no noisy nets, no distributional output.
Exploration via epsilon-greedy with linear decay.
"""

import torch
from tianshou.policy import DQNPolicy
from .q_network import StandardQNetwork


def make_double_agent(cfg, n_actions: int, n_stack: int, device: str, action_space):
    """
    Args:
        cfg:          double_dqn section of config.yaml
        n_actions:    number of discrete actions
        n_stack:      number of stacked frames
        device:       'cpu' or 'cuda'
        action_space: gymnasium action space

    Returns:
        policy:       DQNPolicy (is_double=True) ready for training
    """
    network = StandardQNetwork(
        n_actions  = n_actions,
        n_stack    = n_stack,
        hidden_dim = cfg["hidden_dim"],
    ).to(device)

    optim = torch.optim.Adam(network.parameters(), lr=cfg["lr"])

    policy = DQNPolicy(
        model              = network,
        optim              = optim,
        discount_factor    = cfg["gamma"],
        estimation_step    = cfg["n_step"],
        target_update_freq = cfg["target_update_freq"],
        is_double          = True,
        action_space       = action_space,
    )

    return policy


def make_train_fn(policy, cfg, total_epochs: int, step_per_epoch: int):
    """Returns a train_fn callback that linearly decays epsilon."""
    eps_start      = cfg["eps_train_start"]
    eps_end        = cfg["eps_train_end"]
    eps_decay_steps = cfg["eps_decay_steps"]

    def train_fn(epoch, env_step):
        eps = max(eps_end, eps_start - env_step * (eps_start - eps_end) / eps_decay_steps)
        policy.set_eps(eps)

    return train_fn
