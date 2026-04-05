"""Neural network components for PPO policies."""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TinyMapExtractor(BaseFeaturesExtractor):
    """Custom CNN for processing the 20x20 tilemap efficiently."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class MultiInputMapExtractor(BaseFeaturesExtractor):
    """Dict observation extractor: CNN over minimap image + scalar features."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        feature_space = observation_space["features"]
        n_input_channels = image_space.shape[0]
        n_scalar_features = feature_space.shape[0]

        # Image branch mirrors the minimap CNN used by previous image-only policies.
        self.image_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_img = th.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.image_cnn(sample_img).shape[1]

        # Fused head consumes [cnn_latent, scalar_features] for policy/value backbones.
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_scalar_features, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        image_latent = self.image_cnn(observations["image"])
        scalar_features = observations["features"]
        return self.linear(th.cat([image_latent, scalar_features], dim=1))
