"""
Scans RAM to find the kart angle address.

Strategy: accelerate straight, snapshot RAM, steer hard left for 60 frames,
snapshot again. Any byte that changed significantly and consistently is a
candidate for angle. Repeats with right turn to confirm direction.

Run:
    docker run -it --rm -v "$(pwd):/code" snes3 \
      -c "cp /code/snes9x_libretro_fixed.so /opt/conda/envs/dev/lib/python3.10/site-packages/retro/cores/snes9x_libretro.so && \
          /opt/conda/envs/dev/bin/python /code/scripts/setup_game.py && \
          /opt/conda/envs/dev/bin/python /code/scripts/scan_angle.py"
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

def run_and_snapshot(phase_actions, n_frames):
    env = retro.make(game=GAME, state=STATE, render_mode=None)
    env.reset()
    # accelerate to get moving
    for _ in range(80):
        env.step(ACCEL)
    snap_before = np.frombuffer(env.get_ram(), dtype=np.uint8).copy()
    for _ in range(n_frames):
        env.step(phase_actions)
    snap_after = np.frombuffer(env.get_ram(), dtype=np.uint8).copy()
    env.close()
    return snap_before, snap_after

print("Running LEFT turn scan...")
before_l, after_l = run_and_snapshot(ACCEL_LEFT, 80)
diff_l = after_l.astype(int) - before_l.astype(int)

print("Running RIGHT turn scan...")
before_r, after_r = run_and_snapshot(ACCEL_RIGHT, 80)
diff_r = after_r.astype(int) - before_r.astype(int)

# Candidates: changed during both turns, in opposite directions
print("\nCandidates (changed in opposite directions for left vs right turn):")
print(f"  {'addr':>6}  {'hex':>6}  {'left_diff':>10}  {'right_diff':>11}  {'left_val':>9}  {'right_val':>10}")
print("-" * 65)

candidates = []
for i in range(len(diff_l)):
    dl = diff_l[i]
    dr = diff_r[i]
    # Must change meaningfully (>5) and in opposite directions
    if abs(dl) > 5 and abs(dr) > 5 and (dl * dr < 0):
        candidates.append((i, dl, dr, int(after_l[i]), int(after_r[i])))

candidates.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
for addr, dl, dr, vl, vr in candidates[:30]:
    print(f"  {addr:>6}  {addr:#06x}  {dl:>+10}  {dr:>+11}  {vl:>9}  {vr:>10}")

if not candidates:
    print("  No clean candidates found. Printing largest changes during left turn:")
    large = sorted(enumerate(diff_l), key=lambda x: abs(x[1]), reverse=True)[:20]
    for addr, dl in large:
        print(f"  {addr:>6}  {addr:#06x}  left_diff={dl:>+6}  right_diff={diff_r[addr]:>+6}  left_val={int(after_l[addr]):>4}  right_val={int(after_r[addr]):>4}")
