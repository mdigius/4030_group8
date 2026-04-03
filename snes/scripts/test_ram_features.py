"""
Verifies vx, vy (velocity components) and kart status RAM addresses.

  0x1022 (4130) — X velocity
  0x1024 (4132) — Y velocity
  0x10A0 (4256) — kart status (0=ground, 2=jump, 4=fallen)

Accelerates straight, then steers left/right, printing all three each step.
vx/vy should change direction when steering. Status should stay 0 on ground.

Run:
    docker run -it --rm -v "$(pwd):/code" snes3 \
      -c "cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && \
          /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && \
          /opt/conda/envs/dev/bin/python /code/scripts/test_ram_features.py"
"""

import sys
import numpy as np
sys.path.insert(0, "/code")
import retro

GAME  = "SuperMarioKart-Snes"
STATE = "mario2"

def make_buttons(indices):
    act = np.zeros(12, dtype=np.int8)
    for i in indices:
        act[i] = 1
    return act

ACCEL       = make_buttons([0])
ACCEL_LEFT  = make_buttons([0, 6])
ACCEL_RIGHT = make_buttons([0, 7])

def read_features(env):
    ram    = env.get_ram()
    speed  = int(np.frombuffer(ram[4330:4332], dtype=np.dtype("<i2"))[0])
    vx     = int(np.frombuffer(ram[4130:4132], dtype=np.dtype("<i2"))[0])
    vy     = int(np.frombuffer(ram[4132:4134], dtype=np.dtype("<i2"))[0])
    status = int(np.frombuffer(ram[4256:4257], dtype=np.dtype("|u1"))[0])
    return speed, vx, vy, status

env = retro.make(game=GAME, state=STATE, render_mode=None)
env.reset()

print(f"  {'frame':>5}  {'phase':<10}  {'speed':>7}  {'vx':>7}  {'vy':>7}  {'status':>7}")
print("-" * 55)

frame = 0
phases = [
    ("ACCEL",    ACCEL,       60),
    ("LEFT",     ACCEL_LEFT,  60),
    ("RIGHT",    ACCEL_RIGHT, 60),
    ("STRAIGHT", ACCEL,       40),
]

for phase_name, action, n in phases:
    for i in range(n):
        env.step(action)
        frame += 1
        if i % 10 == 0:
            speed, vx, vy, status = read_features(env)
            print(f"  {frame:>5}  {phase_name:<10}  {speed:>7}  {vx:>7}  {vy:>7}  {status:>7}")

env.close()
print("\nvx/vy should change when steering left vs right.")
print("status should be 0 while on ground, non-zero if fallen.")
