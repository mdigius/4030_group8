"""
Q-network architectures for Mario Kart RL agents.

  StandardQNetwork  — Double DQN  (plain CNN → Q values)
  DuelingQNetwork   — Dueling DQN (V + A streams)
  PPOActorNet       — PPO actor   (shared backbone → action logits)
  PPOCriticNet      — PPO critic  (shared backbone → state value)
"""

import numpy as np
import torch
import torch.nn as nn

N_STACK = 4 * 3   # 4 stacked RGB frames = 12 input channels; must match env.py N_STACK * N_CHANNELS_PER_FRAME


# ── Shared conv feature extractor ────────────────────────────────────────────

class _ConvBackbone(nn.Module):
    """
    CNN sized for 96×72 SNES frames (cropped top half, resized for speed).

    Conv output size for (H=72, W=96):
      After Conv1 (k=8,s=4): H=(72-8)/4+1=17,  W=(96-8)/4+1=23
      After Conv2 (k=4,s=2): H=(17-4)/2+1=7,   W=(23-4)/2+1=10
      After Conv3 (k=3,s=1): H=(7-3)/1+1=5,    W=(10-3)/1+1=8
      Flat: 64 * 5 * 8 = 2560
    """
    FRAME_H = 72
    FRAME_W = 96
    OUT_DIM = 64 * 5 * 8  # = 2560

    def __init__(self, n_stack=N_STACK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,      64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,      64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs):
        """obs: (batch, H, W, n_stack*3) uint8 → float32 normalised."""
        if hasattr(obs, "pixels"):
            obs = obs.pixels
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2) / 255.0   # → (batch, n_stack, H, W)
        return self.net(obs)


# ── Standard Q-Network (Double DQN) ──────────────────────────────────────────

class StandardQNetwork(nn.Module):
    """Plain CNN Q-network for Double DQN. Outputs Q(s,a) per action."""

    def __init__(self, n_actions: int, n_stack: int = N_STACK, hidden_dim: int = 256):
        super().__init__()
        self.n_actions  = n_actions
        self.output_dim = hidden_dim

        self.backbone = _ConvBackbone(n_stack)
        self.net = nn.Sequential(
            nn.Linear(_ConvBackbone.OUT_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs, state=None, **kwargs):
        return self.net(self.backbone(obs)), state


# ── Dueling Q-Network ─────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN (Wang et al., 2016).
    Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
    """

    def __init__(self, n_actions: int, n_stack: int = N_STACK, hidden_dim: int = 256):
        super().__init__()
        self.n_actions  = n_actions
        self.output_dim = hidden_dim

        self.backbone = _ConvBackbone(n_stack)
        self.fc = nn.Sequential(nn.Linear(_ConvBackbone.OUT_DIM, hidden_dim), nn.ReLU())

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs, state=None, **kwargs):
        features  = self.fc(self.backbone(obs))
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q, state


# ── PPO Networks (shared backbone, orthogonal init) ───────────────────────────

def _ortho(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Apply orthogonal init to all Conv2d and Linear layers."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class _PPOBackbone(nn.Module):
    """
    Shared CNN + feature-fusion trunk used by both PPO actor and critic.

    Processes two inputs:
      pixels   — (B, 72, 96, 12) uint8 → CNN → 2560-dim  (4 stacked RGB frames)
      features — (B, N_FEATURES) float32 → Linear → 32-dim

    Both are concatenated then projected to hidden_dim via a single FC layer.
    Orthogonal init throughout for stable early gradients.
    """

    FEAT_DIM = 64  # feature embedding size — increased for 5-dim input

    def __init__(self, n_stack: int = N_STACK, hidden_dim: int = 256, n_features: int = 3):
        super().__init__()
        self.output_dim = hidden_dim
        self.cnn        = _ortho(_ConvBackbone(n_stack))
        self.feat_embed = _ortho(nn.Sequential(
            nn.Linear(n_features, self.FEAT_DIM), nn.ReLU(),
            nn.Linear(self.FEAT_DIM, self.FEAT_DIM), nn.ReLU(),
        ))
        # Three-stage compression: 2560+64 → 1024 → 512 → hidden_dim
        self.fc = _ortho(nn.Sequential(
            nn.Linear(_ConvBackbone.OUT_DIM + self.FEAT_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim), nn.ReLU(),
        ))

    def forward(self, obs):
        # obs is a Tianshou Batch with .pixels and .features when using Dict obs space
        if hasattr(obs, "pixels"):
            pixels   = obs.pixels
            features = obs.features
        else:
            # fallback: plain array (e.g. during manual testing)
            pixels   = obs
            features = torch.zeros(pixels.shape[0], 3, dtype=torch.float32,
                                   device=pixels.device if isinstance(pixels, torch.Tensor) else "cpu")

        if isinstance(pixels, np.ndarray):
            pixels = torch.as_tensor(pixels, dtype=torch.float32)
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features, dtype=torch.float32)

        # Ensure both on same device
        features = features.to(pixels.device)

        cnn_out  = self.cnn(pixels)                  # (B, 2560)
        feat_out = self.feat_embed(features)          # (B, 32)
        fused    = torch.cat([cnn_out, feat_out], dim=1)  # (B, 2592)
        return self.fc(fused)                         # (B, hidden_dim)


class PPOActorNet(nn.Module):
    """
    PPO actor — outputs raw action logits (no softmax).
    Tianshou's PPOPolicy applies Categorical(logits=...) internally.

    Policy head uses gain=0.01 → near-uniform initial policy, which
    prevents the agent from committing to one action before any learning.
    """

    def __init__(self, n_actions: int, n_stack: int = N_STACK, hidden_dim: int = 256, n_features: int = 3):
        super().__init__()
        self.backbone = _PPOBackbone(n_stack, hidden_dim, n_features)
        self.head = nn.Linear(hidden_dim, n_actions)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, state=None, **kwargs):
        return self.head(self.backbone(obs)), state


class PPOCriticNet(nn.Module):
    """
    PPO critic — outputs scalar V(s).

    Shares the CNN backbone with the actor so both heads benefit from the
    same learned visual features. The value head uses gain=1.0 (standard).
    Only the head parameters need to be added to the optimizer separately
    — the backbone is already covered via the actor.
    """

    def __init__(self, backbone: _PPOBackbone, hidden_dim: int = 256):
        super().__init__()
        self.backbone = backbone                        # shared reference
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, **kwargs):
        return self.head(self.backbone(obs))            # (B, 1)
