"""
Creates a minimal stable-retro integration for Super Mario Kart (SNES)
and copies the ROM into the correct data directory.
"""
import hashlib
import json
import os
import shutil
import retro
import glob
import gzip

ROM_SRC = "/code/Super Mario Kart (USA).sfc"

# Find stable-retro's data directory
retro_data_dir = os.path.join(os.path.dirname(retro.__file__), "data", "stable")
game_dir = os.path.join(retro_data_dir, "SuperMarioKart-Snes")
os.makedirs(game_dir, exist_ok=True)
print(f"Game dir: {game_dir}")

# Compute ROM SHA1
with open(ROM_SRC, "rb") as f:
    rom_bytes = f.read()
sha1 = hashlib.sha1(rom_bytes).hexdigest()
print(f"ROM SHA1: {sha1}")

# Copy ROM
rom_dst = os.path.join(game_dir, "rom.sfc")
shutil.copy2(ROM_SRC, rom_dst)
print(f"Copied ROM to {rom_dst}")

# rom.sha
with open(os.path.join(game_dir, "rom.sha"), "w") as f:
    f.write(sha1)

# metadata.json
metadata = {"default_state": "TimeTrial"}
with open(os.path.join(game_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# data.json — verified RAM variables for Super Mario Kart (USA, SNES)
# Addresses are decimal offsets into WRAM ($7E0000 + offset on the SNES bus).
# All types use standard numpy dtype strings supported by the current libretro core.
# <d2 (BCD) is intentionally avoided — it crashes the updated snes9x core.
data = {
    "info": {
        # ── Position ──────────────────────────────────────────────────────────
        "kart1_X":            {"address": 136,  "type": "<i2"},  # $7E0088 east coord
        "kart1_Y":            {"address": 140,  "type": "<i2"},  # $7E008C south coord
        "kart1_direction":    {"address": 149,  "type": "|u1"},  # $7E0095 facing angle
        # ── Motion ───────────────────────────────────────────────────────────
        "kart1_speed":        {"address": 4330, "type": "<i2"},  # $7E10EA scalar speed
        "vel_east":           {"address": 4130, "type": "<i2"},  # $7E1022 east velocity
        "vel_south":          {"address": 4132, "type": "<i2"},  # $7E1024 south velocity
        # ── Race progress ─────────────────────────────────────────────────────
        "current_checkpoint": {"address": 4316, "type": "|u1"},  # $7E10DC checkpoint index
        "totalCheckpoints":   {"address": 328,  "type": "|u1"},  # $7E0148 checkpoints/lap
        "lap":                {"address": 4289, "type": "|u1"},  # $7E10C1 raw lap (127=lap0)
        "rank":               {"address": 4160, "type": "|u1"},  # $7E1040 race position
        # ── State flags ──────────────────────────────────────────────────────
        "isTurnedAround":     {"address": 267,  "type": "|u1"},  # $7E010B facing-wrong-way flag
        # ── Timer (raw u2 — avoids BCD crash; divide by 100 for seconds) ─────
        "currMiliSec":        {"address": 256,  "type": "<u2"},  # $7E0100
        "currSec":            {"address": 258,  "type": "<u2"},  # $7E0102
        "currMin":            {"address": 260,  "type": "<u2"},  # $7E0104
    }
}
with open(os.path.join(game_dir, "data.json"), "w") as f:
    json.dump(data, f, indent=2)

# scenario.json — minimal (no variable-based rewards, agent learns from observations)
scenario = {
    "reward": {"variables": {}},
    "done": {"variables": {}}
}
with open(os.path.join(game_dir, "scenario.json"), "w") as f:
    json.dump(scenario, f, indent=2)

# Copy any .state files from /code into the game directory
for state_file in glob.glob("/code/*.state"):
    dst = os.path.join(game_dir, os.path.basename(state_file))
    with open(state_file, "rb") as f:
        file_data = f.read()

    # Gzip if not already gzipped
    if file_data.startswith(b'\x1f\x8b'):
        with open(dst, "wb") as f:
            f.write(file_data)
    else:
        with gzip.open(dst, "wb") as f:
            f.write(file_data)

    print(f"Copied {os.path.basename(state_file)} ({len(file_data)} bytes) to {dst}")

print("Integration files written.")
