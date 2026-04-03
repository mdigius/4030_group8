"""
Main training script — reads config.yaml to select and train the agent.

Usage:
    python train.py                  # uses agent from config.yaml
    python train.py --agent ppo
    python train.py --agent dueling
    python train.py --agent double_dqn
"""

import os
import time
import argparse
import yaml
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger  # type: ignore[import-untyped]

from env import make_env, RetroToGym

PPO_AGENT   = "ppo"
OFF_POLICY  = {"dueling", "double_dqn"}


# ── Console colours ──────────────────────────────────────────────────────────

class C:
    RESET   = '\033[0m';  BOLD    = '\033[1m';  DIM     = '\033[2m'
    RED     = '\033[91m'; YELLOW  = '\033[93m'; GREEN   = '\033[92m'
    BLUE    = '\033[94m'; CYAN    = '\033[96m'; WHITE   = '\033[97m'
    GRAY    = '\033[90m'; MAGENTA = '\033[95m'

def _cprint(color, prefix, msg):
    print(f"{color}{C.BOLD}{prefix}{C.RESET} {msg}")

def log_info(label, value):
    print(f"  {C.GRAY}{label:<24}{C.RESET}{C.WHITE}{value}{C.RESET}")

def log_ok(msg):    _cprint(C.GREEN,   "  ✔", msg)
def log_warn(msg):  _cprint(C.YELLOW,  "  !", msg)
def log_save(msg):  _cprint(C.CYAN,    "  ↓", msg)
def log_done(msg):  _cprint(C.MAGENTA, "  ★", msg)

def print_header(title):
    W = 64
    print(f"\n{C.CYAN}{C.BOLD}╔{'═'*(W-2)}╗{C.RESET}")
    pad = (W - 2 - len(title)) // 2
    print(f"{C.CYAN}{C.BOLD}║{' '*pad}{title}{' '*(W-2-pad-len(title))}║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}╚{'═'*(W-2)}╝{C.RESET}\n")

def print_section(title):
    print(f"\n{C.BLUE}{C.BOLD}  ┌─ {title} {C.DIM}{'─'*max(0,52-len(title))}{C.RESET}")

def epoch_bar(epoch, total, env_step):
    pct    = 100 * epoch / total
    bar_w  = 30
    filled = int(bar_w * epoch / total)
    bar = f"{C.GREEN}{'█'*filled}{C.GRAY}{'░'*(bar_w-filled)}{C.RESET}"
    print(f"\n{C.BOLD}  Epoch {C.CYAN}{epoch:>3}{C.RESET}{C.BOLD}/{total}"
          f"  [{bar}{C.BOLD}]  {C.YELLOW}{pct:5.1f}%{C.RESET}"
          f"  {C.GRAY}env steps: {env_step:,}{C.RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["ppo", "dueling", "double_dqn"],
                        help="Override agent selection from config.yaml")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    agent_name = args.agent or cfg["agent"]
    agent_cfg  = cfg[agent_name]
    env_cfg    = cfg["env"]
    train_cfg  = cfg["training"]

    checkpoint_dir = "/code/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"policy_{agent_name}.pth")

    log_path = train_cfg["log_path"]
    os.makedirs(log_path, exist_ok=True)

    print_header(f"Super Mario Kart — {agent_name.upper()} Training")

    cv2.setNumThreads(0)
    # allow PyTorch to use all available cores for network forward/backward passes
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # ── Environments ──────────────────────────────────────────────────────────
    print_section("Environment")
    num_envs = env_cfg["num_envs"]

    grayscale = env_cfg.get("grayscale", False)

    def _make_env():
        cv2.setNumThreads(0)
        torch.set_num_threads(1)  # keep env subprocesses single-threaded to avoid contention
        return make_env(env_cfg["game"], env_cfg["state"], render_mode=None, grayscale=grayscale)

    train_envs = SubprocVectorEnv([_make_env] * num_envs)
    n_actions  = RetroToGym.N_ACTIONS
    n_channels = 1 if grayscale else 3
    n_stack    = RetroToGym.N_STACK * n_channels

    step_per_collect = agent_cfg["step_per_collect"]
    batch_size       = agent_cfg["batch_size"]

    log_info("Agent",           f"{C.CYAN}{agent_name.upper()}{C.RESET}")
    log_info("Parallel envs",   f"{C.GREEN}{num_envs}{C.RESET}")
    log_info("Observation",     f"84×84 × {n_stack} stacked frames")
    log_info("Actions",         f"{n_actions} discrete combos")
    log_info("Steps/collect",   f"{step_per_collect:,}")
    log_info("Epochs",          f"{train_cfg['epochs']}  ×  {train_cfg['step_per_epoch']:,} steps")

    # ── Network & policy ──────────────────────────────────────────────────────
    print_section("Network")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info("Device", f"{C.YELLOW}{device.upper()}{C.RESET}")

    action_space = train_envs.action_space[0]
    total_steps  = train_cfg["epochs"] * train_cfg["step_per_epoch"]

    if agent_name == PPO_AGENT:
        from agents.ppo_agent import make_ppo_agent, make_train_fn
        policy   = make_ppo_agent(agent_cfg, n_actions, n_stack, device, action_space,
                                   n_features=env_cfg.get("n_features", 3))
        train_fn = make_train_fn(policy, agent_cfg, total_steps)
        total_params = sum(p.numel() for p in policy.actor.parameters()) \
                     + sum(p.numel() for p in policy.critic.head.parameters())
        log_info("Parameters",      f"{total_params:,}")
        log_info("GAE λ",           f"{agent_cfg['gae_lambda']}")
        log_info("Clip ε",          f"{agent_cfg['eps_clip']}")
        log_info("Entropy coef",    f"{agent_cfg['ent_coef']} → {agent_cfg['ent_coef_end']}")
        log_info("Repeat/collect",  f"{agent_cfg['repeat_per_collect']}  PPO epochs per rollout")
    elif agent_name == "dueling":
        from agents.dueling_dqn_agent import make_dueling_agent, make_train_fn
        policy   = make_dueling_agent(agent_cfg, n_actions, n_stack, device, action_space)
        train_fn = make_train_fn(policy, agent_cfg, train_cfg["epochs"], train_cfg["step_per_epoch"])
        total_params = sum(p.numel() for p in policy.model.parameters())
        log_info("Parameters",    f"{total_params:,}")
        log_info("N-step",        f"{agent_cfg['n_step']}")
    else:
        from agents.double_dqn_agent import make_double_agent, make_train_fn
        policy   = make_double_agent(agent_cfg, n_actions, n_stack, device, action_space)
        train_fn = make_train_fn(policy, agent_cfg, train_cfg["epochs"], train_cfg["step_per_epoch"])
        total_params = sum(p.numel() for p in policy.model.parameters())
        log_info("Parameters",    f"{total_params:,}")
        log_info("N-step",        f"{agent_cfg['n_step']}")

    log_info("Learning rate",   f"{agent_cfg['lr']}")
    log_info("Gamma",           f"{agent_cfg['gamma']}")
    log_info("Checkpoint",      save_path)

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    print_section("Checkpoint")
    if os.path.exists(save_path):
        try:
            policy.load_state_dict(torch.load(save_path, map_location=device))
            log_ok(f"Resumed from  {C.CYAN}{save_path}{C.RESET}")
        except Exception as e:
            log_warn(f"Load failed ({e}) — starting fresh")
    else:
        log_warn(f"No checkpoint found — starting fresh")

    # ── Collector & buffer ────────────────────────────────────────────────────
    if agent_name in OFF_POLICY:
        buf_size = train_cfg["buffer_size"]
        buf = VectorReplayBuffer(total_size=buf_size, buffer_num=num_envs)
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
        log_info("Replay buffer", f"Uniform  size={buf_size:,}")
    else:
        # PPO is on-policy — no persistent buffer; Tianshou manages the rollout
        train_collector = Collector(policy, train_envs)
        log_info("Replay buffer", "None (on-policy)")

    # ── Logger ────────────────────────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(log_path, agent_name))
    logger = TensorboardLogger(writer)

    # ── Best-model tracking ───────────────────────────────────────────────────
    # Wrap collector.collect so we can capture per-epoch episode rewards without
    # needing a separate test collector.
    best_path      = os.path.join(checkpoint_dir, f"policy_{agent_name}_best.pth")
    best_rew_path  = os.path.join(checkpoint_dir, f"policy_{agent_name}_best_rew.txt")
    _prior_best    = float("-inf")
    if os.path.exists(best_rew_path):
        try:
            _prior_best = float(open(best_rew_path).read().strip())
            log_ok(f"Loaded previous best reward: {C.GREEN}{_prior_best:.2f}{C.RESET}")
        except Exception:
            pass
    _best_rew      = [_prior_best]
    _latest_rew    = [float("-inf")]
    _orig_collect  = train_collector.collect

    def _tracked_collect(*args, **kwargs):
        result = _orig_collect(*args, **kwargs)
        if result.get("n/ep", 0) > 0:
            _latest_rew[0] = result["rew"]
        return result

    train_collector.collect = _tracked_collect

    # ── Callbacks ─────────────────────────────────────────────────────────────
    _t0 = [time.time()]

    def _train_fn(epoch, env_step):
        _t0[0] = time.time()
        epoch_bar(epoch, train_cfg["epochs"], env_step)
        train_fn(epoch, env_step)

    def save_checkpoint(epoch, env_step, gradient_step):  # noqa: ARG001
        torch.save(policy.state_dict(), save_path)
        elapsed = time.time() - _t0[0]
        log_save(f"Saved  {C.CYAN}{save_path}{C.RESET}"
                 f"  {C.GRAY}(epoch {epoch}, {elapsed:.1f}s){C.RESET}")
        # Save best model separately whenever a new reward high-score is reached
        if _latest_rew[0] > _best_rew[0]:
            _best_rew[0] = _latest_rew[0]
            torch.save(policy.state_dict(), best_path)
            open(best_rew_path, "w").write(str(_best_rew[0]))
            log_ok(f"New best  {C.GREEN}{_best_rew[0]:.2f}{C.RESET}"
                   f"  → {C.CYAN}{best_path}{C.RESET}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print_section("Training")
    print()

    if agent_name == PPO_AGENT:
        from tianshou.trainer import onpolicy_trainer
        result = onpolicy_trainer(
            policy             = policy,
            train_collector    = train_collector,
            test_collector     = None,
            max_epoch          = train_cfg["epochs"],
            step_per_epoch     = train_cfg["step_per_epoch"],
            repeat_per_collect = agent_cfg["repeat_per_collect"],
            episode_per_test   = 0,
            batch_size         = batch_size,
            step_per_collect   = step_per_collect,
            train_fn           = _train_fn,
            save_checkpoint_fn = save_checkpoint,
            logger             = logger,
            verbose            = True,
        )
    else:
        from tianshou.trainer import offpolicy_trainer
        result = offpolicy_trainer(
            policy             = policy,
            train_collector    = train_collector,
            test_collector     = None,
            max_epoch          = train_cfg["epochs"],
            step_per_epoch     = train_cfg["step_per_epoch"],
            step_per_collect   = step_per_collect,
            update_per_step    = agent_cfg["update_per_step"],
            episode_per_test   = 0,
            batch_size         = batch_size,
            train_fn           = _train_fn,
            save_checkpoint_fn = save_checkpoint,
            logger             = logger,
            verbose            = True,
        )

    torch.save(policy.state_dict(), save_path)
    print()
    log_done(f"Training complete — policy saved to {C.CYAN}{save_path}{C.RESET}")
    log_info("Best reward", str(result.get("best_reward", "n/a")))

    train_envs.close()


if __name__ == "__main__":
    main()
