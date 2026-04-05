"""Callbacks used by PPO training."""

from stable_baselines3.common.callbacks import BaseCallback


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class BeautifulCallback(BaseCallback):
    """Colorized rollout progress and checkpoint saving."""

    def __init__(self, total_timesteps, save_path, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.save_path = save_path
        self.best_mean_reward = -float("inf")

    def _on_rollout_end(self) -> None:
        pct = 100 * self.num_timesteps / self.total_timesteps
        bar_w = 30
        filled = int(bar_w * self.num_timesteps / max(self.total_timesteps, 1))
        bar = f"{C.GREEN}{'█' * filled}{C.GRAY}{'░' * (bar_w - filled)}{C.RESET}"

        if len(self.model.ep_info_buffer) > 0:
            ep_rew = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer]) / len(
                self.model.ep_info_buffer
            )
        else:
            ep_rew = 0.0

        print(
            f"\n{C.BOLD}  Rollout {C.CYAN}{(self.num_timesteps // self.model.n_steps)}{C.RESET}"
            f"  [{bar}{C.BOLD}]  {C.YELLOW}{pct:5.1f}%{C.RESET}"
            f"  {C.GRAY}steps: {self.num_timesteps:,}{C.RESET}"
            f"  {C.GREEN}rew: {ep_rew:.1f}{C.RESET}"
        )

        self.model.save(self.save_path)

        if len(self.model.ep_info_buffer) > 0 and ep_rew > self.best_mean_reward:
            self.best_mean_reward = ep_rew
            best_path = self.save_path.replace(".zip", "_best.zip")
            self.model.save(best_path)
            print(
                f"{C.GREEN}{C.BOLD}  ✔{C.RESET} New best training reward: "
                f"{C.GREEN}{ep_rew:.1f}{C.RESET} (Saved best_model)"
            )

    def _on_step(self) -> bool:
        return True
