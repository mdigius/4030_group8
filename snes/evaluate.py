import os
import yaml
import argparse
import cv2
import numpy as np
import torch
from stable_baselines3 import PPO

from env import make_env

# We must import the custom extractor so torch can load the policy correctly,
# or PPO.load handles it internally. Since SB3 handles it if it's saved in the zip,
# we just need to ensure the architecture matches.
from agents.ppo.network import MultiInputMapExtractor
from agents.dueling.network import DuelingQNetwork


def _run_ppo_eval(env, save_path: str, episodes: int):
    print(f"Loading model from {save_path}...")
    try:
        # Strip .zip to avoid SB3 appending an extra .zip
        load_target = save_path[:-4] if save_path.endswith(".zip") else save_path
        model = PPO.load(
            load_target,
            custom_objects={"features_extractor_class": MultiInputMapExtractor},
        )
    except Exception as e:
        print(f"No valid model found or load failed: {e}. Untrained agent fallback.")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=dict(features_extractor_class=MultiInputMapExtractor),
        )

    all_rewards = []
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        num_steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            num_steps += 1
            print(f"\rStep: {num_steps} Reward: {total_reward:.2f}", end="")

        print(f"\nEpisode {ep} finished with Reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

    print(
        f"\nAverage Reward over {episodes} episodes: {sum(all_rewards) / len(all_rewards):.2f}"
    )


def _dueling_action(net: DuelingQNetwork, obs, device: str) -> int:
    # Network accepts Dict obs and returns (Q-values, state).
    with torch.no_grad():
        q_values, _ = net(obs)
        if isinstance(q_values, np.ndarray):
            q_values = torch.from_numpy(q_values)
        q_values = q_values.to(device)
        action = int(torch.argmax(q_values, dim=1).item())
    return action


def _run_dueling_eval(env, env_cfg: dict, episodes: int):
    obs_space = env.observation_space
    image_shape = obs_space["image"].shape
    n_features = int(obs_space["features"].shape[0])
    n_actions = env.action_space.n
    hidden_dim = env_cfg["dueling"]["hidden_dim"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DuelingQNetwork(
        image_shape=image_shape,
        n_features=n_features,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
    ).to(device)

    best_path = "/code/checkpoints/policy_dueling_dict_best.pth"
    latest_path = "/code/checkpoints/policy_dueling_dict.pth"
    load_path = best_path if os.path.exists(best_path) else latest_path

    print(f"Loading model from {load_path}...")
    if os.path.exists(load_path):
        net.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print("No valid dueling checkpoint found. Random action fallback.")
        net = None

    all_rewards = []
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        num_steps = 0

        while not done:
            if net is None:
                action = int(env.action_space.sample())
            else:
                action = _dueling_action(net, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            num_steps += 1
            print(f"\rStep: {num_steps} Reward: {total_reward:.2f}", end="")

        print(f"\nEpisode {ep} finished with Reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

    print(
        f"\nAverage Reward over {episodes} episodes: {sum(all_rewards) / len(all_rewards):.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent = cfg.get("agent", "ppo")
    speed_reward = env_cfg.get("speed_reward", False)
    reward_scale = env_cfg.get("reward_scale", 1.0)
    suffix = "_speed" if speed_reward else "_no_speed"
    suffix = f"{suffix}_dict"
    save_path = f"/code/checkpoints/policy_ppo{suffix}_best.zip"

    env = make_env(
        env_cfg["game"],
        env_cfg["state"],
        render_mode="human",
        grayscale=False,
        speed_reward=speed_reward,
        reward_scale=reward_scale,
    )

    if agent == "ppo":
        _run_ppo_eval(env, save_path, args.episodes)
    elif agent == "dueling":
        _run_dueling_eval(env, cfg, args.episodes)
    else:
        raise ValueError(f"Unsupported agent '{agent}'. Supported: ppo, dueling")

    env.close()


if __name__ == "__main__":
    main()
