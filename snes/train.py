"""Training entrypoint wrapper.

Dispatches to the configured agent trainer package.
"""

import os
import yaml

from agents.ppo.trainer import main as ppo_main
from agents.dueling.trainer import main as dueling_main


def main():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    agent = cfg.get("agent", "ppo")
    if agent == "ppo":
        ppo_main()
        return
    if agent == "dueling":
        dueling_main()
        return

    raise ValueError(f"Unsupported agent '{agent}'. Supported: ppo, dueling")


if __name__ == "__main__":
    main()
