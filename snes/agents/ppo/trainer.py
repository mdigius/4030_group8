"""PPO training orchestration."""

import os
import yaml
import argparse
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env import make_env, RetroToGym
from agents.ppo import BeautifulCallback, MultiInputMapExtractor


# ── Console colours ──────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    MAGENTA = "\033[95m"


def _cprint(color, prefix, msg):
    print(f"{color}{C.BOLD}{prefix}{C.RESET} {msg}")


def log_info(label, value):
    print(f"  {C.GRAY}{label:<24}{C.RESET}{C.WHITE}{value}{C.RESET}")


def log_ok(msg):
    _cprint(C.GREEN, "  ✔", msg)


def log_warn(msg):
    _cprint(C.YELLOW, "  !", msg)


def log_done(msg):
    _cprint(C.MAGENTA, "  ★", msg)


def print_header(title):
    W = 64
    print(f"\n{C.CYAN}{C.BOLD}╔{'═'*(W-2)}╗{C.RESET}")
    pad = (W - 2 - len(title)) // 2
    print(f"{C.CYAN}{C.BOLD}║{' '*pad}{title}{' '*(W-2-pad-len(title))}║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}╚{'═'*(W-2)}╝{C.RESET}\n")


def print_section(title):
    print(f"\n{C.BLUE}{C.BOLD}  ┌─ {title} {C.DIM}{'─'*max(0,52-len(title))}{C.RESET}")


def main():
    parser = argparse.ArgumentParser()
    _args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "../../config.yaml")) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    num_envs = env_cfg["num_envs"]
    speed_reward = env_cfg.get("speed_reward", False)
    reward_scale = env_cfg.get("reward_scale", 1.0)

    checkpoint_dir = "/code/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = "_speed" if speed_reward else "_no_speed"
    # Keep dict-input checkpoints isolated from legacy image-only runs.
    suffix = f"{suffix}_dict"
    save_path = os.path.join(checkpoint_dir, f"policy_ppo{suffix}.zip")
    tensorboard_log = f"/code/logs/ppo{suffix}"

    print_header("Super Mario Kart — PPO2 Training")

    def _make_env():
        env_instance = make_env(
            env_cfg["game"],
            env_cfg["state"],
            render_mode=None,
            grayscale=False,
            speed_reward=speed_reward,
            reward_scale=reward_scale,
        )
        return Monitor(env_instance)

    print_section("Environment")
    env = SubprocVecEnv([_make_env for _ in range(num_envs)])

    n_stack = RetroToGym.N_STACK * 3
    n_actions = RetroToGym.N_ACTIONS
    log_info("Agent", f"{C.CYAN}PPO2 (Stable Baselines 3){C.RESET}")
    log_info("Parallel envs", f"{C.GREEN}{num_envs}{C.RESET}")
    log_info(
        "Observation", f"Dict(image=20×20×{n_stack}, features={RetroToGym.N_FEATURES})"
    )
    log_info("Actions", f"{n_actions} discrete combos")

    policy_kwargs = dict(
        features_extractor_class=MultiInputMapExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    n_steps = 256
    batch_size = (n_steps * num_envs) // 4
    n_epochs = 4
    learning_rate = 2.5e-4
    total_timesteps = 5_000_000

    print_section("Network")
    device = "cuda" if th.cuda.is_available() else "cpu"
    log_info("Device", f"{C.YELLOW}{device.upper()}{C.RESET}")
    log_info("Learning rate", f"{learning_rate}")
    log_info("Steps/rollout", f"{n_steps}")
    log_info("Checkpoint", save_path)

    print_section("Checkpoint")

    # Strip extension so SB3 does not append an extra .zip during load.
    load_target = save_path[:-4] if save_path.endswith(".zip") else save_path

    if os.path.exists(save_path):
        model = PPO.load(
            load_target, env=env, custom_objects={"tensorboard_log": tensorboard_log}
        )
        log_ok(f"Resumed from {C.CYAN}{save_path}{C.RESET}")
        model.tensorboard_log = tensorboard_log
    else:
        log_warn("No checkpoint found — starting fresh")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=tensorboard_log,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.02,
        )

    pretty_callback = BeautifulCallback(
        total_timesteps=total_timesteps, save_path=save_path
    )

    print_section("Training")
    print()

    try:
        model.learn(total_timesteps=total_timesteps, callback=pretty_callback)
    except KeyboardInterrupt:
        print("\n")
        log_warn("Manual interrupt, saving model...")

    model.save(save_path)
    print()
    log_done(f"Training complete — policy saved to {C.CYAN}{save_path}{C.RESET}")
    try:
        env.close()
    except (BrokenPipeError, EOFError, OSError) as e:
        log_warn(f"VecEnv close warning (safe to ignore after interrupt): {e}")
