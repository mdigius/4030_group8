"""
Drives around the track and builds a correct tile-byte -> surface-type mapping
by correlating the tilemap value under the kart with the actual surface RAM byte.

Run inside Docker:
    python scripts/build_physics_map.py

Outputs: /code/physics_map.json (overwrites existing)
"""

import sys
import json
import numpy as np

sys.path.insert(0, "/code")
import retro

env = retro.make(game="SuperMarioKart-Snes", state="mario2", render_mode="rgb_array")
obs, _ = env.reset()

# Button helpers
BTN_B = np.zeros(12, dtype=np.int8)
BTN_B[0] = 1  # accelerate

BTN_B_LEFT = np.zeros(12, dtype=np.int8)
BTN_B_LEFT[0] = 1
BTN_B_LEFT[6] = 1

BTN_B_RIGHT = np.zeros(12, dtype=np.int8)
BTN_B_RIGHT[0] = 1
BTN_B_RIGHT[7] = 1

BTN_B_L = np.zeros(12, dtype=np.int8)
BTN_B_L[0] = 1
BTN_B_L[10] = 1

# Mapping: tile_byte -> { surface_value: count }
tile_surface_counts = {}

def read_state(ram):
    kart_x = int(np.frombuffer(ram[136:138], dtype=np.dtype("<i2"))[0])
    kart_y = int(np.frombuffer(ram[140:142], dtype=np.dtype("<i2"))[0])
    surface = int(np.frombuffer(ram[4270:4271], dtype=np.dtype("|u1"))[0])
    game_mode = int(np.frombuffer(ram[181:182], dtype=np.dtype("|u1"))[0])
    return kart_x, kart_y, surface, game_mode

def sample_tile(ram):
    kart_x, kart_y, surface, game_mode = read_state(ram)
    if game_mode != 0x1C:
        return

    tx = max(0, min(127, kart_x // 8))
    ty = max(0, min(127, kart_y // 8))

    tilemap = np.frombuffer(ram[0x8000:0x8000 + 128*128], dtype=np.uint8).reshape((128, 128))
    tile_byte = int(tilemap[ty, tx])

    if tile_byte not in tile_surface_counts:
        tile_surface_counts[tile_byte] = {}
    if surface not in tile_surface_counts[tile_byte]:
        tile_surface_counts[tile_byte][surface] = 0
    tile_surface_counts[tile_byte][surface] += 1

# Phase 1: Wait through countdown
print("Phase 1: Waiting through countdown...")
for _ in range(220):
    obs, _, _, _, _ = env.step(BTN_B)

# Phase 2: Drive various patterns to cover the track
# Pattern: go straight, turn left, turn right, drift, etc.
patterns = [
    ("Straight",     BTN_B,       800),
    ("Left",         BTN_B_LEFT,  400),
    ("Straight",     BTN_B,       600),
    ("Right",        BTN_B_RIGHT, 400),
    ("Straight",     BTN_B,       800),
    ("Right",        BTN_B_RIGHT, 300),
    ("Straight",     BTN_B,       600),
    ("Left",         BTN_B_LEFT,  300),
    ("Straight",     BTN_B,       800),
    ("Left",         BTN_B_LEFT,  500),
    ("Straight",     BTN_B,       800),
    ("Right",        BTN_B_RIGHT, 500),
    ("Drift Left",   BTN_B_L,     200),
    ("Straight",     BTN_B,       600),
    ("Left",         BTN_B_LEFT,  600),
    ("Right",        BTN_B_RIGHT, 600),
    ("Straight",     BTN_B,       1000),
    ("Left",         BTN_B_LEFT,  800),
    ("Straight",     BTN_B,       1000),
    ("Right",        BTN_B_RIGHT, 800),
]

total_frames = 0
for name, action, frames in patterns:
    print(f"Phase 2: {name} for {frames} frames...")
    for _ in range(frames):
        obs, _, terminated, truncated, _ = env.step(action)
        ram = env.get_ram()
        sample_tile(ram)
        total_frames += 1
        if terminated or truncated:
            # Reset and skip countdown
            obs, _ = env.reset()
            for _ in range(220):
                obs, _, _, _, _ = env.step(BTN_B)

print(f"\nTotal frames sampled: {total_frames}")
print(f"Unique tile bytes seen: {len(tile_surface_counts)}")

# Build the final mapping: for each tile byte, pick the most common surface
physics_map = {}
surface_names = {64: "road", 84: "grass", 128: "wall", 40: "fall", 32: "deep", 34: "water"}

print("\n--- Tile Byte -> Surface Mapping ---")
print(f"{'Tile':>6} | {'Surface':>10} | {'Counts'}")
print("-" * 60)

for tile_byte in sorted(tile_surface_counts.keys()):
    counts = tile_surface_counts[tile_byte]
    best_surface = max(counts, key=counts.get)
    total = sum(counts.values())
    confidence = counts[best_surface] / total * 100

    physics_map[str(tile_byte)] = best_surface

    counts_str = ", ".join(f"{surface_names.get(s, str(s))}={c}" for s, c in sorted(counts.items()))
    print(f"{tile_byte:>6} | {surface_names.get(best_surface, str(best_surface)):>10} | {counts_str}  ({confidence:.0f}%)")

# Fill unmapped tile bytes as grass (conservative default)
for i in range(256):
    if str(i) not in physics_map:
        physics_map[str(i)] = 84

# Save
out_path = "/code/physics_map.json"
with open(out_path, "w") as f:
    json.dump(physics_map, f, sort_keys=True)

print(f"\nSaved to {out_path}")

# Print summary
surface_summary = {}
for tile, surface in physics_map.items():
    surface_summary[surface] = surface_summary.get(surface, 0) + 1
print("\nSurface distribution:")
for surface, count in sorted(surface_summary.items()):
    print(f"  {surface_names.get(surface, str(surface)):>6}: {count} tile types")

# Render full track PNG with the new mapping
import cv2

ram = env.get_ram()
tilemap = np.frombuffer(ram[0x8000:0x8000 + 128*128], dtype=np.uint8).reshape((128, 128))

tile_to_surface = np.zeros(256, dtype=np.uint8)
for t, s in physics_map.items():
    tile_to_surface[int(t)] = s

surface_map = tile_to_surface[tilemap]
img = np.zeros((128, 128, 3), dtype=np.uint8)
img[surface_map == 64]  = [0, 200, 0]    # road  -> green
img[surface_map == 84]  = [100, 100, 0]  # grass -> dim yellow
img[surface_map == 128] = [0, 0, 200]    # wall  -> red
img[(surface_map != 64) & (surface_map != 84) & (surface_map != 128)] = [40, 40, 40]

# Mark kart
kart_x = int(np.frombuffer(ram[136:138], dtype=np.dtype("<i2"))[0])
kart_y = int(np.frombuffer(ram[140:142], dtype=np.dtype("<i2"))[0])
tx = max(0, min(127, kart_x // 8))
ty = max(0, min(127, kart_y // 8))
img[max(0,ty-1):ty+2, max(0,tx-1):tx+2] = [255, 0, 0]

img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
big = cv2.resize(img_bgr, (128*4, 128*4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("/code/minimap_new_physics.png", big)
print(f"\nSaved /code/minimap_new_physics.png (full track with new mapping)")

env.close()
