#!/bin/bash
# Run the trained model with live visualization via XQuartz.
set -e

# Start XQuartz and allow connections from localhost
open -a XQuartz
sleep 2
export DISPLAY=:0
/opt/X11/bin/xhost + >/dev/null 2>&1 || true

CORE_INSTALL='cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so'
PYTHON='/opt/conda/envs/dev/bin/python'
PIP_DEPS="${PYTHON} -m pip install pyyaml -q"

docker run -it --rm \
  -v "$(pwd):/code" \
  --env DISPLAY=host.docker.internal:0 \
  --env QT_X11_NO_MITSHM=1 \
  --name snes-eval \
  snes3 \
  -c "${CORE_INSTALL} && ${PIP_DEPS} && ${PYTHON} /code/scripts/setup_game.py && ${PYTHON} /code/evaluate.py $@"
