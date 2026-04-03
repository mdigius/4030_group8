"""
Optimised PPO agent factory.

Implements Proximal Policy Optimisation (Schulman et al., 2017) with every
standard improvement from the "Implementation Matters" literature:

  1. Shared CNN backbone   — actor & critic share visual features, one forward
                             pass per env step, better gradient signal
  2. Orthogonal init       — CNN + FC layers use gain=√2; policy head gain=0.01
                             (near-uniform initial policy); value head gain=1.0
  3. GAE (λ=0.95)          — lower-variance advantage estimates via exponential
                             weighting of n-step returns
  4. PPO clip (ε=0.2)      — prevents destructively large policy updates
  5. Value function clip    — clip V loss to the same ε as the policy, stops
                             value from straying too far each update
  6. Entropy bonus          — encourages exploration; decays linearly to
                             ent_coef_end so the agent commits as it learns
  7. Gradient clipping      — global norm clipped at 0.5, prevents gradient
                             explosions common with RNNs and deep CNNs
  8. Linear LR annealing    — lr decays to 0 over the full training budget;
                             fine-tunes policy without over-shooting at the end
  9. Advantage normalisation — zero-mean / unit-std per mini-batch; removes
                             scale dependence from reward shaping choices
 10. Large rollout buffer   — collect 512+ steps before each update so PPO
                             mini-batches see diverse temporal context

Unlike DQN, PPO is on-policy: collected rollouts are used for `repeat_per_collect`
gradient passes then discarded. No replay buffer needed.
"""

import torch
import torch.nn as nn
from tianshou.policy import PPOPolicy
from .q_network import PPOActorNet, PPOCriticNet


def make_ppo_agent(cfg, n_actions: int, n_stack: int, device: str, action_space, n_features: int = 3):
    """
    Args:
        cfg:          ppo section of config.yaml
        n_actions:    number of discrete actions
        n_stack:      number of stacked frames (temporal context)
        device:       'cpu' or 'cuda'
        action_space: gymnasium action space

    Returns:
        policy:  PPOPolicy ready for onpolicy_trainer
    """
    hidden_dim = cfg["hidden_dim"]

    actor  = PPOActorNet(n_actions, n_stack, hidden_dim, n_features).to(device)
    critic = PPOCriticNet(actor.backbone, hidden_dim).to(device)

    # Shared backbone params come from actor; only add critic HEAD params
    # separately to avoid double-counting in the optimiser.
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.head.parameters()),
        lr     = cfg["lr"],
        eps    = 1e-5,   # tighter epsilon → more stable updates than default 1e-8
    )

    policy = PPOPolicy(
        actor              = actor,
        critic             = critic,
        optim              = optim,
        dist_fn            = lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor    = cfg["gamma"],
        gae_lambda         = cfg["gae_lambda"],
        max_grad_norm      = cfg["max_grad_norm"],
        vf_coef            = cfg["vf_coef"],
        ent_coef           = cfg["ent_coef"],
        eps_clip           = cfg["eps_clip"],
        value_clip         = True,   # clip V loss to eps_clip, same as policy
        advantage_normalization = True,
        action_space       = action_space,
    )

    return policy


def make_train_fn(policy, cfg, total_steps: int):
    """
    Returns a train_fn callback for onpolicy_trainer.

    Applies two annealing schedules each epoch:
      • Linear LR decay:      lr → 0 over total_steps env steps
      • Linear entropy decay: ent_coef → ent_coef_end over total_steps

    Both are proven to improve final performance on continuous-action and
    discrete game environments.
    """
    lr_start      = cfg["lr"]
    ent_start     = cfg["ent_coef"]
    ent_end       = cfg["ent_coef_end"]
    optim         = policy.optim

    def train_fn(epoch, env_step):
        frac = min(1.0, env_step / max(total_steps, 1))

        # LR annealing
        new_lr = lr_start * (1.0 - frac)
        for pg in optim.param_groups:
            pg["lr"] = max(new_lr, 1e-7)

        # Entropy annealing
        policy.ent_coef = ent_start + frac * (ent_end - ent_start)

    return train_fn
