"""
Evaluate a trained agent with live visualisation.

Usage:
    python evaluate.py                     # uses agent from config.yaml
    python evaluate.py --agent ppo
    python evaluate.py --agent dueling
    python evaluate.py --agent double_dqn
    python evaluate.py --episodes 5
    python evaluate.py --recent
"""

import os
import argparse
import yaml
import cv2
import torch
from tianshou.data import Batch

from env import make_env, RetroToGym


# ── Console colours ──────────────────────────────────────────────────────────

class C:
    RESET   = '\033[0m';  BOLD    = '\033[1m'
    GREEN   = '\033[92m'; YELLOW  = '\033[93m'; CYAN    = '\033[96m'
    GRAY    = '\033[90m'; MAGENTA = '\033[95m'; WHITE   = '\033[97m'
    RED     = '\033[91m'

def log_info(label, value):
    print(f"  {C.GRAY}{label:<22}{C.RESET}{C.WHITE}{value}{C.RESET}")

def log_ok(msg):    print(f"{C.GREEN}{C.BOLD}  ✔{C.RESET} {msg}")
def log_warn(msg):  print(f"{C.YELLOW}{C.BOLD}  !{C.RESET} {msg}")
def log_done(msg):  print(f"{C.MAGENTA}{C.BOLD}  ★{C.RESET} {msg}")

def print_header(title):
    W = 64
    print(f"\n{C.CYAN}{C.BOLD}╔{'═'*(W-2)}╗{C.RESET}")
    pad = (W - 2 - len(title)) // 2
    print(f"{C.CYAN}{C.BOLD}║{' '*pad}{title}{' '*(W-2-pad-len(title))}║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}╚{'═'*(W-2)}╝{C.RESET}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    choices=["ppo", "dueling", "double_dqn"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--recent",   action="store_true", help="Evaluate the most recent model instead of the best")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    agent_name = args.agent or cfg["agent"]
    agent_cfg  = cfg[agent_name]
    env_cfg    = cfg["env"]

    recent_path = f"/code/checkpoints/policy_{agent_name}.pth"
    best_path   = f"/code/checkpoints/policy_{agent_name}_best.pth"

    if args.recent:
        save_path = recent_path
    else:
        save_path = best_path if os.path.exists(best_path) else recent_path

    print_header(f"Super Mario Kart — {agent_name.upper()} Evaluation")

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    n_actions = RetroToGym.N_ACTIONS
    grayscale = env_cfg.get("grayscale", False)
    n_channels = 1 if grayscale else 3
    n_stack   = RetroToGym.N_STACK * n_channels

    log_info("Agent",    agent_name.upper())
    log_info("Device",   device.upper())
    log_info("Episodes", str(args.episodes))
    log_info("Model",    save_path)
    print()

    # ── Build env ─────────────────────────────────────────────────────────────
    env = make_env(env_cfg["game"], env_cfg["state"], render_mode="human", grayscale=grayscale)
    action_space = env.action_space

    # ── Build policy (same architecture as training) ──────────────────────────
    if agent_name == "ppo":
        from agents.ppo_agent import make_ppo_agent
        policy = make_ppo_agent(agent_cfg, n_actions, n_stack, device, action_space,
                                n_features=env_cfg.get("n_features", 3))
    elif agent_name == "dueling":
        from agents.dueling_dqn_agent import make_dueling_agent
        policy = make_dueling_agent(agent_cfg, n_actions, n_stack, device, action_space)
    else:
        from agents.double_dqn_agent import make_double_agent
        policy = make_double_agent(agent_cfg, n_actions, n_stack, device, action_space)

    # ── Load weights ──────────────────────────────────────────────────────────
    if os.path.exists(save_path):
        try:
            policy.load_state_dict(torch.load(save_path, map_location=device))
            log_ok(f"Loaded  {C.CYAN}{save_path}{C.RESET}")
        except Exception as e:
            log_warn(f"Could not load weights ({e}) — running untrained agent")
    else:
        log_warn(f"No checkpoint at {save_path} — running untrained agent")

    policy.eval()

    # DQN agents use epsilon-greedy — set to 0 for greedy eval
    # PPO samples from its learned distribution (no eps to set)
    if agent_name in ("dueling", "double_dqn"):
        policy.set_eps(0.0)

    print()

    # ── Episode loop ──────────────────────────────────────────────────────────
    all_rewards = []
    all_steps   = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done         = False
        total_reward = 0.0
        steps        = 0

        print(f"{C.BOLD}  Episode {C.CYAN}{ep}{C.RESET}{C.BOLD}/{args.episodes}{C.RESET}")

        while not done:
            batch = Batch(obs=[obs], info=[info])
            with torch.no_grad():
                result = policy(batch)
            action = int(result.act[0])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1
            done = terminated or truncated

            rew_color = C.GREEN if total_reward > 0 else C.RED
            print(f"\r  {C.GRAY}step:{C.RESET} {C.WHITE}{steps:>6,}{C.RESET}"
                  f"  {C.GRAY}reward:{C.RESET} {rew_color}{total_reward:>9.2f}{C.RESET}  ",
                  end="", flush=True)

        print()  # newline after episode ends

        all_rewards.append(total_reward)
        all_steps.append(steps)

        rew_color = C.GREEN if total_reward > 0 else C.RED
        print(f"  {C.GRAY}final →{C.RESET} {rew_color}{total_reward:8.2f}{C.RESET}"
              f"  {C.GRAY}steps:{C.RESET} {C.WHITE}{steps:,}{C.RESET}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    log_done("Evaluation complete")
    log_info("Avg reward", f"{sum(all_rewards)/len(all_rewards):.2f}")
    log_info("Avg steps",  f"{sum(all_steps)//len(all_steps):,}")
    log_info("Best reward",f"{max(all_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
