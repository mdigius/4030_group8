"""
Headless test — verifies kart angle RAM address 0x10AA (decimal 4266).

Starts from mario2 savestate, accelerates straight for 60 frames,
then steers left for 120 frames, printing the angle each step.
A working angle address should show steady change while turning.

Run:
    docker run -it --rm -v "$(pwd):/code" snes3 \
      -c "cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && \
          /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && \
          /opt/conda/envs/dev/bin/python /code/scripts/test_angle.py"
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

# Button order: B Y SELECT START UP DOWN LEFT RIGHT A X L R
ACCEL       = make_buttons([0])       # B
ACCEL_LEFT  = make_buttons([0, 6])    # B + LEFT
ACCEL_RIGHT = make_buttons([0, 7])    # B + RIGHT

def read_byte(env, addr):
    ram = env.get_ram()
    return int(np.frombuffer(ram[addr:addr+1], dtype=np.dtype("|u1"))[0])

env = retro.make(game=GAME, state=STATE, render_mode=None)
env.reset()

print("Accelerating straight for 80 frames...")
for _ in range(80):
    env.step(ACCEL)

print(f"\n{'phase':<12} {'frame':>5}  {'addr 4266 (0x10AA)':>18}  {'addr 4267 (0x10AB)':>18}  {'speed':>7}")
print("-" * 70)

def print_row(phase, frame):
    a1 = read_byte(env, 4266)
    a2 = read_byte(env, 4267)
    ram = env.get_ram()
    spd = int(np.frombuffer(ram[4330:4332], dtype=np.dtype("<i2"))[0])
    print(f"{phase:<12} {frame:>5}  {a1:>18}  {a2:>18}  {spd:>7}")

print("Steering LEFT for 60 frames:")
for i in range(60):
    env.step(ACCEL_LEFT)
    if i % 10 == 0:
        print_row("LEFT", i)

print("\nSteering RIGHT for 60 frames:")
for i in range(60):
    env.step(ACCEL_RIGHT)
    if i % 10 == 0:
        print_row("RIGHT", i)

print("\nStraight for 30 frames:")
for i in range(30):
    env.step(ACCEL)
    if i % 10 == 0:
        print_row("STRAIGHT", i)

env.close()
print("\nDone. The correct angle address should change consistently when steering LEFT/RIGHT.")
print("It should decrease turning left and increase turning right (or vice versa).")
