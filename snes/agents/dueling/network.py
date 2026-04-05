"""Dueling DQN network for Dict observations (image + scalar features)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn


class _MapBackbone(nn.Module):
    """CNN over minimap image plus MLP over scalar features."""

    def __init__(
        self, image_shape: Tuple[int, int, int], n_features: int, hidden_dim: int
    ):
        super().__init__()
        c, h, w = image_shape
        self.image_cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros((1, c, h, w), dtype=torch.float32)
            n_flatten = self.image_cnn(sample).shape[1]

        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten + 64, hidden_dim),
            nn.ReLU(),
        )

    def _extract(self, obs: Any):
        if isinstance(obs, dict):
            return obs["image"], obs["features"]
        if hasattr(obs, "image") and hasattr(obs, "features"):
            return obs.image, obs.features
        if hasattr(obs, "obs"):
            return self._extract(obs.obs)
        raise TypeError("Unsupported observation type for DuelingQNetwork")

    def _to_tensor(
        self, x: Any, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        if torch.is_tensor(x):
            return x.to(dtype=dtype, device=device)
        return torch.as_tensor(x, dtype=dtype, device=device)

    def forward(self, obs: Any) -> torch.Tensor:
        image, features = self._extract(obs)
        device = next(self.parameters()).device
        image_t = self._to_tensor(image, torch.float32, device)
        features_t = self._to_tensor(features, torch.float32, device)

        if image_t.ndim == 3:
            image_t = image_t.unsqueeze(0)
        if features_t.ndim == 1:
            features_t = features_t.unsqueeze(0)

        image_t = image_t / 255.0
        img_latent = self.image_cnn(image_t)
        feat_latent = self.feature_mlp(features_t)
        return self.fusion(torch.cat([img_latent, feat_latent], dim=1))


class DuelingQNetwork(nn.Module):
    """Dueling head: Q(s,a)=V(s)+A(s,a)-mean(A)."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        n_features: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone = _MapBackbone(image_shape, n_features, hidden_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: Any, state=None, **kwargs):
        h = self.backbone(obs)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q, state
