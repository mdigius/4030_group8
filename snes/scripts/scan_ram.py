"""
RAM address verifier for Super Mario Kart (SNES).

Runs the emulator and prints the value at each candidate address across three phases:
  1. Idle  — 120 frames, no buttons pressed (countdown)
  2. Accel — 120 frames, B held (accelerating straight)
  3. Left  — 120 frames, B+LEFT (turning left)

Output shows min/max/final so you can confirm which addresses are live.
Also does a diff scan to find ALL addresses that changed between idle and accel.
"""

import os, struct
import numpy as np
import retro

# ── Load emulator via retro.make ─────────────────────────────────────────────
env = retro.make(game="SuperMarioKart-Snes", state="TimeTrial", render_mode="rgb_array")
env.reset()

RAM_LEN = len(env.get_ram())
print(f"RAM size: {RAM_LEN} bytes ({RAM_LEN // 1024} KB)\n")

# ── Candidate addresses ───────────────────────────────────────────────────────
# (name, address, numpy_dtype)
CANDIDATES = [
    ("kart1_X",            136,  "<i2"),
    ("kart1_Y",            140,  "<i2"),
    ("kart1_direction",    149,  "|u1"),
    ("isTurnedAround",     267,  "|u1"),
    ("currMiliSec",        256,  "<u2"),
    ("currSec",            258,  "<u2"),
    ("currMin",            260,  "<u2"),
    ("current_checkpoint", 4316, "|u1"),
    ("lap",                4289, "|u1"),
    ("rank",               4160, "|u1"),
    ("kart1_speed",        4330, "<i2"),
    ("vel_east",           4130, "<i2"),
    ("vel_south",          4132, "<i2"),
    ("totalCheckpoints",   328,  "|u1"),
    # New candidates for off-track detection
    ("surface",            4270, "|u1"),
    ("KartState",          4112, "|u1"),
    ("DrivingMode",        58,   "|u1"),
]

def read_addr(ram, addr, dtype):
    size = np.dtype(dtype).itemsize
    if addr + size > len(ram):
        return None
    return int(np.frombuffer(bytes(ram[addr:addr+size]), dtype=dtype)[0])

# Button layout: B Y SEL START UP DOWN LEFT RIGHT A X L R (index 0–11)
BTN_NONE = np.zeros(12, dtype=np.int8)
BTN_B    = np.array([1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int8)   # B (accel)
BTN_BL   = np.array([1,0,0,0,0,0,1,0,0,0,0,0], dtype=np.int8)   # B + LEFT

def run_phase(buttons, frames, label):
    stats = {name: [] for name, _, _ in CANDIDATES}
    for _ in range(frames):
        env.step(buttons)
        ram = env.get_ram()
        for name, addr, dtype in CANDIDATES:
            v = read_addr(ram, addr, dtype)
            if v is not None:
                stats[name].append(v)
    print(f"\n{'─'*62}")
    print(f"  Phase: {label} ({frames} frames)")
    print(f"{'─'*62}")
    print(f"  {'Name':<22} {'addr':>6}  {'min':>8}  {'max':>8}  {'final':>8}  changed?")
    print(f"  {'-'*22} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  --------")
    for name, addr, dtype in CANDIDATES:
        vals = stats[name]
        if not vals:
            print(f"  {name:<22} {addr:>6}  OUT OF RANGE")
            continue
        mn, mx, final = min(vals), max(vals), vals[-1]
        flag = "YES <--" if mn != mx else "no"
        print(f"  {name:<22} {addr:>6}  {mn:>8}  {mx:>8}  {final:>8}  {flag}")
    return stats

def ram_snapshot():
    return bytes(env.get_ram())

# ── Run phases ────────────────────────────────────────────────────────────────
run_phase(BTN_NONE, 220, "IDLE — no buttons (Lakitu countdown, ~180 frames)")
snap_idle  = ram_snapshot()

run_phase(BTN_B,    180, "ACCEL — B held (straight)")
snap_accel = ram_snapshot()

run_phase(BTN_BL,   180, "TURN LEFT — B+LEFT held")

# Drive hard right into the wall/grass to capture off-track surface values
BTN_BR = np.array([1,0,0,0,0,0,0,1,0,0,0,0], dtype=np.int8)
run_phase(BTN_BR,   180, "HARD RIGHT — into grass/wall")

# ── Diff scan: addresses that changed idle→accel ──────────────────────────────
print(f"\n{'─'*62}")
print("  DIFF SCAN: all addresses that changed between idle→accel")
print("  (2-byte signed reads, first 8192 bytes of WRAM, first 60 hits)")
print(f"{'─'*62}")
hits = []
for addr in range(0, min(8192, RAM_LEN - 1)):
    v_idle  = struct.unpack_from("<h", snap_idle,  addr)[0]
    v_accel = struct.unpack_from("<h", snap_accel, addr)[0]
    if v_idle != v_accel:
        hits.append((addr, v_idle, v_accel))

for addr, v_before, v_after in hits[:60]:
    print(f"  addr {addr:>5} (0x{addr:04X})  idle={v_before:>6}  accel={v_after:>6}  delta={v_after-v_before:>+6}")

if not hits:
    print("  No differences found — RAM may not be updating correctly.")

print(f"\nDone. Total diff hits in first 8KB: {len(hits)}")
env.close()
