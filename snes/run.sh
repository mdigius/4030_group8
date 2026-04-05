#!/bin/bash
# Headless training — no display required.
set -e

CORE_INSTALL='cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so'
PYTHON='/opt/conda/envs/dev/bin/python'
PIP_DEPS="${PYTHON} -m pip install pyyaml stable-baselines3[extra] -q"

# Patch Tianshou segtree: replace the assertion with a safe clip so that
# PrioritizedVectorReplayBuffer doesn't crash on fp-precision edge cases.
PATCH_SEGTREE="${PYTHON} -c \"
import sys
f = '/opt/conda/envs/dev/lib/python3.10/site-packages/tianshou/data/utils/segtree.py'
old = '        assert np.all(value >= 0.0) and np.all(value < self._value[1])'
new = '        value = np.clip(np.asarray(value, dtype=np.float64), 0.0, float(self._value[1]) * (1.0 - 1e-9) if float(self._value[1]) > 0 else 0.0)'
src = open(f).read()
if old in src:
    open(f, 'w').write(src.replace(old, new))
    print('[patch] segtree.py fixed')
else:
    print('[patch] segtree.py already patched')
\""

if [ "$1" = "--scan" ]; then
  CMD="${CORE_INSTALL} && ${PIP_DEPS} && ${PYTHON} /code/scripts/setup_game.py && ${PYTHON} /code/scripts/scan_ram.py"
elif [ "$1" = "--r" ]; then
  rm -f checkpoints/policy_*.zip checkpoints/policy_*_best.zip checkpoints/policy_*_best_rew.txt
  rm -rf logs/
  echo "Checkpoints, best reward, and logs cleared."
  CMD="${CORE_INSTALL} && ${PIP_DEPS} && ${PATCH_SEGTREE} && ${PYTHON} /code/scripts/setup_game.py && ${PYTHON} /code/train.py"
else
  CMD="${CORE_INSTALL} && ${PIP_DEPS} && ${PATCH_SEGTREE} && ${PYTHON} /code/scripts/setup_game.py && ${PYTHON} /code/train.py"
fi

docker run -it --rm \
  -v "$(pwd):/code" \
  --name snes-train \
  snes3 \
  -c "${CMD}"
