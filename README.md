# Super Mario Kart RL (stable-retro + PPO)

This repository trains and evaluates a PPO agent for SNES Super Mario Kart using a minimap-based observation.

The current agent stack uses:
- Dict observations from env.py:
    - image: stacked minimap tensor (C, H, W)
    - features: scalar RAM-derived vector
- PPO with MultiInputPolicy
- Custom extractor in agents/ppo/network.py

## Project layout

- env.py: Gymnasium environment wrapper, reward shaping, minimap extraction, Dict observations
- agents/ppo/network.py: PPO feature extractors (image-only and image+features)
- agents/ppo/callbacks.py: rollout progress and checkpoint callback
- agents/ppo/trainer.py: PPO training orchestration
- train.py: thin training entrypoint
- evaluate.py: policy evaluation runner
- debug/show_cnn_input.py: exports the minimap frames used by the policy

## Requirements

- Docker
- A built image named snes3 (used by run.sh and eval.sh)
- ROM/state assets already present in this repo

## Common commands

### Train

Run training in Docker:

```bash
./run.sh
```

Reset checkpoints/logs and train fresh:

```bash
./run.sh --r
```

### Evaluate

Run evaluation with rendering:

```bash
./eval.sh
```

Run a specific number of episodes:

```bash
./eval.sh --episodes 1
```

### Minimap debug export

Regenerate PNGs showing what the CNN sees:

```bash
docker run -it --rm \
    -v "$(pwd):/code" \
    --name snes-minimap-test \
    snes3 \
    -c 'cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && /opt/conda/envs/dev/bin/python -m pip install pyyaml stable-baselines3[extra] -q && /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && /opt/conda/envs/dev/bin/python /code/debug/show_cnn_input.py'
```

Outputs are written in repo root:
- minimap_frame_0.png .. minimap_frame_3.png
- minimap_stack.png
- minimap_full_track.png

## Notes on checkpoints

- Dict-observation PPO checkpoints use the _dict suffix to avoid architecture mismatch with older models.
- Current naming example:
    - /code/checkpoints/policy_ppo_no_speed_dict.zip
    - /code/checkpoints/policy_ppo_no_speed_dict_best.zip

## macOS display notes

If GUI rendering fails on macOS, configure XQuartz and allow indirect GLX:

```bash
defaults write org.xquartz.X11 enable_iglx -bool true
```

Then restart XQuartz (and if needed, restart macOS).
