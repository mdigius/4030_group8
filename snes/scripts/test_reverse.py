"""
Headless test — figures out which button combos reverse the kart.

Accelerates for 60 frames to get moving, then tries each candidate
reverse combo for 120 frames and prints speed each step.
No display required — runs fully headless.

Run:
    docker run -it --rm -v "$(pwd):/code" snes3 \
      -c "cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && \
          /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && \
          /opt/conda/envs/dev/bin/python /code/scripts/test_reverse.py"
"""

import sys
import numpy as np
sys.path.insert(0, "/code")

import retro

GAME  = "SuperMarioKart-Snes"
STATE = "mario2"

# Button order: B Y SELECT START UP DOWN LEFT RIGHT A X L R
#               0 1    2     3   4    5    6     7  8 9 10 11

def make_buttons(indices):
    act = np.zeros(12, dtype=np.int8)
    for i in indices:
        act[i] = 1
    return act

ACCEL          = make_buttons([0])        # B
BRAKE          = make_buttons([1])        # Y
BRAKE_LEFT     = make_buttons([1, 6])     # Y + LEFT
BRAKE_RIGHT    = make_buttons([1, 7])     # Y + RIGHT
NOTHING        = make_buttons([])         # no buttons

CANDIDATES = [
    ("BRAKE only",     BRAKE),
    ("BRAKE+LEFT",     BRAKE_LEFT),
    ("BRAKE+RIGHT",    BRAKE_RIGHT),
    ("NOTHING",        NOTHING),
]

def read_speed(env):
    ram = env.get_ram()
    return int(np.frombuffer(ram[4330:4332], dtype=np.dtype("<i2"))[0])

def read_surface(env):
    ram = env.get_ram()
    return int(np.frombuffer(ram[4270:4271], dtype=np.dtype("|u1"))[0])

def run_combo(name, combo, accel_frames=80, test_frames=120):
    env = retro.make(game=GAME, state=STATE, render_mode=None)
    env.reset()

    print(f"\n{'='*55}")
    print(f"  Testing: {name}")
    print(f"  Accelerating for {accel_frames} frames...")

    for _ in range(accel_frames):
        env.step(ACCEL)

    speed_after_accel = read_speed(env)
    print(f"  Speed after accel: {speed_after_accel}")
    print(f"  Applying [{name}] for {test_frames} frames:")
    print(f"  {'frame':>6}  {'speed':>7}  {'surface':>8}")

    prev_speed = speed_after_accel
    went_negative = False
    for i in range(test_frames):
        env.step(combo)
        spd = read_speed(env)
        surf_raw = read_surface(env)
        surf_str = {64: "ROAD", 84: "GRASS", 128: "WALL"}.get(surf_raw, f"?{surf_raw}")
        if i % 10 == 0 or (spd < 0 and not went_negative):
            print(f"  {i+1:>6}  {spd:>7}  {surf_str:>8}")
        if spd < 0:
            went_negative = True

    final_speed = read_speed(env)
    if final_speed < 0:
        print(f"  ✔ REVERSES — final speed: {final_speed}")
    elif final_speed == 0:
        print(f"  ~ Stopped but not reversing — final speed: {final_speed}")
    else:
        print(f"  ✘ Still going forward — final speed: {final_speed}")

    env.close()

for name, combo in CANDIDATES:
    run_combo(name, combo)

print("\nDone.")
