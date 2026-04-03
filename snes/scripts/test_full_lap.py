"""
Drives straight for a full lap to verify the lap counter increments correctly.
Prints raw_lap, cur_lap, laps_completed, and checkpoint each step.

Run:
    docker run -it --rm -v "$(pwd):/code" snes3 \
      -c "cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && \
          /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && \
          /opt/conda/envs/dev/bin/python /code/scripts/test_full_lap.py"
"""
import sys
import numpy as np
sys.path.insert(0, "/code")
import retro

GAME     = "SuperMarioKart-Snes"
STATE    = "mario2"
LAP_BASE = 127

def make_buttons(indices):
    act = np.zeros(12, dtype=np.int8)
    for i in indices:
        act[i] = 1
    return act

ACCEL = make_buttons([0])

env = retro.make(game=GAME, state=STATE, render_mode=None)
env.reset()

print(f"  {'frame':>5}  {'raw_lap':>8}  {'cur_lap':>8}  {'cp':>4}")
print("-" * 35)

prev_raw = None
for i in range(5000):
    env.step(ACCEL)
    ram     = env.get_ram()
    raw_lap = int(np.frombuffer(ram[4289:4290], dtype=np.dtype("|u1"))[0])
    cp      = int(np.frombuffer(ram[4316:4317], dtype=np.dtype("|u1"))[0])
    cur_lap = raw_lap - LAP_BASE

    # print whenever lap or cp changes, or every 200 frames
    if raw_lap != prev_raw or i % 200 == 0:
        print(f"  {i+1:>5}  {raw_lap:>8}  {cur_lap:>8}  {cp:>4}")
        prev_raw = raw_lap

env.close()
print("\nDone.")
