"""Dueling DQN training orchestration."""

import os
import argparse
import yaml
import numpy as np
import torch

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer

from env import make_env
from agents.dueling.network import DuelingQNetwork


def main():
    parser = argparse.ArgumentParser()
    _args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "../../config.yaml")) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    dqn_cfg = cfg["dueling"]
    train_cfg = cfg["training"]

    num_envs = env_cfg["num_envs"]
    grayscale = env_cfg.get("grayscale", False)
    speed_reward = env_cfg.get("speed_reward", False)
    reward_scale = env_cfg.get("reward_scale", 1.0)

    checkpoint_dir = "/code/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "policy_dueling_dict.pth")
    best_path = os.path.join(checkpoint_dir, "policy_dueling_dict_best.pth")

    def _mk_env(render_mode=None):
        return make_env(
            env_cfg["game"],
            env_cfg["state"],
            render_mode=render_mode,
            grayscale=grayscale,
            speed_reward=speed_reward,
            reward_scale=reward_scale,
        )

    train_envs = DummyVectorEnv([lambda: _mk_env(None) for _ in range(num_envs)])
    test_envs = DummyVectorEnv([lambda: _mk_env(None)])

    obs_space = train_envs.observation_space
    image_shape = obs_space["image"].shape
    n_features = int(obs_space["features"].shape[0])
    n_actions = train_envs.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DuelingQNetwork(
        image_shape=image_shape,
        n_features=n_features,
        n_actions=n_actions,
        hidden_dim=dqn_cfg["hidden_dim"],
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=dqn_cfg["lr"])

    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=dqn_cfg["gamma"],
        estimation_step=dqn_cfg["n_step"],
        target_update_freq=dqn_cfg["target_update_freq"],
        is_double=True,
        action_space=train_envs.action_space,
    )

    if os.path.exists(save_path):
        state_dict = torch.load(save_path, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Resumed from {save_path}")

    buffer = VectorReplayBuffer(total_size=train_cfg["buffer_size"], buffer_num=num_envs)
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    eps_start = dqn_cfg["eps_train_start"]
    eps_end = dqn_cfg["eps_train_end"]
    eps_decay_steps = max(1, dqn_cfg["eps_decay_steps"])

    def train_fn(epoch, env_step):
        eps = max(eps_end, eps_start - env_step * (eps_start - eps_end) / eps_decay_steps)
        policy.set_eps(float(eps))

    def test_fn(epoch, env_step):
        policy.set_eps(0.01)

    best_mean = -np.inf

    def save_best_fn(pol):
        nonlocal best_mean
        rewards = test_collector.collect(n_episode=1).get("rews", None)
        if rewards is not None and len(rewards) > 0:
            mean_rew = float(np.mean(rewards))
            if mean_rew > best_mean:
                best_mean = mean_rew
                torch.save(pol.state_dict(), best_path)

    print("Starting Dueling DQN training...")
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=train_cfg["epochs"],
        step_per_epoch=train_cfg["step_per_epoch"],
        step_per_collect=dqn_cfg["step_per_collect"],
        episode_per_test=1,
        batch_size=dqn_cfg["batch_size"],
        update_per_step=dqn_cfg["update_per_step"],
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
    )

    torch.save(policy.state_dict(), save_path)
    print(f"Training complete. Saved to {save_path}")
    print(result)


if __name__ == "__main__":
    main()
