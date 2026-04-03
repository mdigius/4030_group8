"""
Verifies candidate angle addresses by watching them change in real time
as the kart steers left, then right, then straight.

Run:
    docker run -it --rm -v "$(pwd):/code" snes3 \
      -c "cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && \
          /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && \
          /opt/conda/envs/dev/bin/python /code/scripts/verify_angle.py"
"""

import sys
import numpy as np
sys.path.insert(0, "/code")
import retro

GAME  = "SuperMarioKart-Snes"
STATE = "mario2"

CANDIDATES = [1604, 1643, 100]  # 0x0644, 0x066b, 0x0064

def make_buttons(indices):
    act = np.zeros(12, dtype=np.int8)
    for i in indices:
        act[i] = 1
    return act

ACCEL       = make_buttons([0])
ACCEL_LEFT  = make_buttons([0, 6])
ACCEL_RIGHT = make_buttons([0, 7])

env = retro.make(game=GAME, state=STATE, render_mode=None)
env.reset()

def read_candidates(env):
    ram = env.get_ram()
    return [int(np.frombuffer(ram[a:a+1], dtype=np.uint8)[0]) for a in CANDIDATES]

header = f"  {'frame':>5}  {'phase':<10}" + "".join(f"  {f'addr {a}':>10}" for a in CANDIDATES)
print(header)
print("-" * len(header))

frame = 0
phases = [
    ("ACCEL",  ACCEL,       60),
    ("LEFT",   ACCEL_LEFT,  80),
    ("RIGHT",  ACCEL_RIGHT, 80),
    ("STRAIGHT", ACCEL,     40),
]

for phase_name, action, n in phases:
    for i in range(n):
        env.step(action)
        frame += 1
        if i % 10 == 0:
            vals = read_candidates(env)
            row = f"  {frame:>5}  {phase_name:<10}" + "".join(f"  {v:>10}" for v in vals)
            print(row)

env.close()
print("\nThe angle address should:")
print("  - Increase steadily while steering LEFT")
print("  - Decrease steadily while steering RIGHT")
print("  - Stabilise while going STRAIGHT")
