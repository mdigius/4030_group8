# Super Mario Kart — Reinforcement Learning

Train and evaluate RL agents on Super Mario Kart (SNES) using [stable-retro](https://github.com/MatPoliquin/stable-retro) inside Docker.

Three agents are implemented: **PPO**, **Dueling DQN**, and **Double DQN**, all configurable via `config.yaml`.

## Requirements

- Docker
- [XQuartz](https://www.xquartz.org/) (macOS only, required for live evaluation display)
- `Super Mario Kart (USA).sfc` ROM (already in this directory)

## Setup

### Build the Docker image

```bash
docker build -t snes3 -f Dockerfile .
```

> **Note (Apple Silicon):** The Dockerfile targets `linux/aarch64` (native ARM64 via Miniforge). No `--platform` flag needed.

## Training

```bash
bash run.sh
```

This mounts the current directory to `/code` inside the container and runs `train.py` using the agent and hyperparameters defined in `config.yaml`.

**Options:**

| Flag | Description |
|------|-------------|
| *(none)* | Resume training from existing checkpoints |
| `--r` | Clear all checkpoints and logs, then start fresh |
| `--scan` | Scan RAM addresses (used for reward shaping development) |

Checkpoints are saved to `checkpoints/` and TensorBoard logs to `logs/`.

To monitor training:

```bash
bash tensorboard.sh
```

## Configuration

Edit `config.yaml` to change the agent, environment, and hyperparameters:

```yaml
agent: ppo  # ppo | dueling | double_dqn

env:
  state: mario2  # TimeTrial | rbroad | MarioCircuit_M | peach | mario2
  num_envs: 8
  grayscale: true
  n_features: 5

training:
  epochs: 300
  step_per_epoch: 10000
```

Each agent (`ppo`, `dueling`, `double_dqn`) has its own hyperparameter section in the config.

## Evaluation

Runs the trained agent with live visualization via XQuartz:

```bash
bash eval.sh
```

**Options:**

```bash
bash eval.sh --agent ppo
bash eval.sh --agent dueling
bash eval.sh --episodes 5
bash eval.sh --recent      # load the most recent checkpoint
```

### Display Setup (macOS)

1. Install [XQuartz](https://www.xquartz.org/)
2. Enable indirect GLX:
   ```bash
   defaults write org.xquartz.X11 enable_iglx -bool true
   ```
3. Restart XQuartz

`eval.sh` handles starting XQuartz and configuring `DISPLAY` automatically.

## Project Structure

```
snes/
├── Dockerfile
├── config.yaml          # agent selection and hyperparameters
├── train.py             # main training script
├── evaluate.py          # evaluation with live display
├── env.py               # stable-retro environment wrapper
├── run.sh               # training launcher (Docker)
├── eval.sh              # evaluation launcher (Docker + XQuartz)
├── tensorboard.sh       # TensorBoard launcher
├── agents/
│   ├── ppo_agent.py
│   ├── dueling_dqn_agent.py
│   ├── double_dqn_agent.py
│   └── q_network.py
├── checkpoints/         # saved model weights
├── logs/                # TensorBoard logs
└── scripts/             # ROM setup and RAM scanning utilities
```
