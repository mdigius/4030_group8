"""
Game environment wrapper for Super Mario Kart (SNES).
Handles frame stacking, action discretisation, reward shaping, and early termination.
RAM addresses verified by scan_ram.py and test_surface2.py.

Observation is a Dict with:
    "image"    — (N_STACK * 3, 20, 20) uint8 stacked minimap channels (C, H, W)
    "features" — (N_FEATURES,) float32: [checkpoint_norm, speed_norm, surface_norm, vx_norm, vy_norm]
"""

import retro
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2


class RetroToGym(gym.Env):

    # Race constants (confirmed by RAM scan)
    LAP_BASE = 128  # reference Lua uses lap - 128
    TOTAL_CHECKPOINTS = 30  # confirmed by totalCheckpoints RAM address
    MAX_LAPS = 5  # matches reference repo — 5 laps per episode

    # Episode limits
    GRACE_STEPS = 60  # ignore harsh crash penalties for initial frames to stabilise PPO
    NO_PROGRESS_STEPS = (
        None  # disabled — reference repo uses time limit + wall accumulation only
    )

    FRAME_SKIP = 4  # deterministic frame skip — matches reference repo
    N_STACK = 4  # frames to stack for temporal context
    N_CHANNELS_PER_FRAME = 3  # default; overridden to 1 per frame when grayscale=True
    N_ACTIONS = 6  # expanded with drift-turning actions
    N_FEATURES = 5  # [checkpoint_norm, speed_norm, surface_norm, vx_norm, vy_norm]
    FRAME_W = 20  # minimap crop width
    FRAME_H = 20  # minimap crop height
    CROP_BOTTOM = 110  # crop minimap — matches reference CutMarioMap

    # Surface values confirmed by test_surface2.py
    SURFACE_ROAD = 64  # clean road
    SURFACE_GRASS = 84  # off-track grass
    SURFACE_WALL = 128  # wall contact (1 frame per bounce)
    SURFACE_FALL = 40  # Rainbow Road void/fall
    SURFACE_DEEP = 32  # deep off-track (reference repo penalises)
    SURFACE_WATER = 34  # water/hazard (reference repo penalises at 0.05 weight)
    GAME_MODE_RACING = 0x1C  # active racing mode
    MAX_EPISODE_TIME = (3, 0)  # (minutes, seconds) — 3-minute time limit

    def __init__(
        self,
        game,
        state=retro.State.NONE,
        render_mode=None,
        grayscale=False,
        speed_reward=False,
        reward_scale=1.0,
    ):
        self.env = retro.make(game=game, state=state, render_mode="rgb_array")
        self.grayscale = grayscale
        self.speed_reward = speed_reward
        self.reward_scale = float(reward_scale)
        self.N_CHANNELS_PER_FRAME = (
            3  # 3 channels for grayscale + colored kart overlays
        )

        self._actions = self._build_actions()
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.N_STACK * self.N_CHANNELS_PER_FRAME,
                        self.FRAME_H,
                        self.FRAME_W,
                    ),
                    dtype=np.uint8,
                ),
                "features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.N_FEATURES,),
                    dtype=np.float32,
                ),
            }
        )
        self._frame_stack = []
        self.render_mode = render_mode
        self._reset_episode_state()

    # ── Action set ────────────────────────────────────────────────────────────
    def _build_actions(self):
        # Button order: B Y SELECT START UP DOWN LEFT RIGHT A X L R
        #               0 1 2      3     4  5    6    7     8 9 10 11
        combos = [
            [0],  # 0  B             — accelerate straight
            [0, 6],  # 1  B + LEFT      — steer left
            [0, 7],  # 2  B + RIGHT     — steer right
            [0, 10],  # 3  B + L         — drift/hop straight
            [0, 6, 10],  # 4  B + LEFT + L  — drift left (for sharp corners)
            [0, 7, 10],  # 5  B + RIGHT + L — drift right (for sharp corners)
        ]
        n = self.env.action_space.n
        actions = []
        for combo in combos:
            act = np.zeros(n, dtype=np.int8)
            for b in combo:
                act[b] = 1
            actions.append(act)
        return actions

    # ── RAM reads ─────────────────────────────────────────────────────────────
    def _read_ram(self):
        ram = self.env.get_ram()
        lap = int(np.frombuffer(ram[4289:4290], dtype=np.dtype("|u1"))[0])
        cp = int(np.frombuffer(ram[4316:4317], dtype=np.dtype("|u1"))[0])
        speed = int(np.frombuffer(ram[4330:4332], dtype=np.dtype("<i2"))[0])
        surface = int(np.frombuffer(ram[4270:4271], dtype=np.dtype("|u1"))[0])
        vx = int(np.frombuffer(ram[4130:4132], dtype=np.dtype("<i2"))[0])
        vy = int(np.frombuffer(ram[4132:4134], dtype=np.dtype("<i2"))[0])
        game_mode = int(np.frombuffer(ram[181:182], dtype=np.dtype("|u1"))[0])  # 0x00B5
        frame = int(
            np.frombuffer(ram[56:58], dtype=np.dtype("<i2"))[0]
        )  # frame counter
        curr_min = int(
            np.frombuffer(ram[260:262], dtype=np.dtype("<u2"))[0]
        )  # in-game minutes
        curr_sec = int(
            np.frombuffer(ram[258:260], dtype=np.dtype("<u2"))[0]
        )  # in-game seconds
        turned = int(
            np.frombuffer(ram[267:268], dtype=np.dtype("|u1"))[0]
        )  # isTurnedAround
        return (
            lap,
            cp,
            speed,
            surface,
            vx,
            vy,
            game_mode,
            frame,
            curr_min,
            curr_sec,
            turned,
        )

    def _make_features(self, cp, speed, surface, vx, vy):
        # surface encoded as 0=road, 0.5=grass/fall, 1.0=wall
        surface_norm = {
            self.SURFACE_ROAD: 0.0,
            self.SURFACE_GRASS: 0.5,
            self.SURFACE_WALL: 1.0,
            self.SURFACE_FALL: 0.5,
        }.get(surface, 0.5)
        # velocity components normalised to [-1, 1] — max observed ~600
        vx_norm = max(-1.0, min(int(vx) / 600.0, 1.0))
        vy_norm = max(-1.0, min(int(vy) / 600.0, 1.0))
        return np.array(
            [
                cp / float(self.TOTAL_CHECKPOINTS - 1),
                max(0, min(int(speed), 1000)) / 1000.0,
                surface_norm,
                vx_norm,
                vy_norm,
            ],
            dtype=np.float32,
        )

    def _reset_episode_state(self):
        self._episode_steps = 0
        self._current_lap = None
        self._laps_completed = 0
        self._prev_total_cp = None  # checkpoint delta tracking
        self._prev_frame = None  # frame counter for reward validation
        self._steps_since_last_cp = 0
        self._last_features = np.zeros(self.N_FEATURES, dtype=np.float32)
        self._prev_surface = None
        # Wall accumulation termination (from esteveste/gym-SuperMarioKart-Snes)
        self._wall_hits = 0
        self._wall_steps = 0

    # ── Reward & termination ──────────────────────────────────────────────────
    # Aligned to esteveste/gym-SuperMarioKart-Snes `script.lua`:
    #   getRewardTrain      = getCheckpointReward() + getExperimentalReward()
    #   getRewardTrainSpeed = getSpeedReward() + checkpoint + experimental
    #   isDoneTrain         = isDone() or isHittingWall()
    def _compute_reward_and_done(
        self,
        lap,
        checkpoint,
        speed,
        surface,
        terminated,
        truncated,
        game_mode,
        frame,
        curr_min,
        curr_sec,
        turned_around,
    ):
        reward = 0.0

        # ── gameMode check — not actively racing ─────────────────────────
        if game_mode != self.GAME_MODE_RACING:
            return reward, True

        # ── 3-minute time limit ──────────────────────────────────────────
        max_min, max_sec = self.MAX_EPISODE_TIME
        if curr_min >= max_min and curr_sec >= max_sec:
            return reward, True

        cur_lap = int(lap) - self.LAP_BASE
        cur_cp = int(checkpoint)

        # ── Optional speed reward (reference: getRewardTrainSpeed*) ─────
        if getattr(self, "speed_reward", False):
            if turned_around == 0x10:
                reward -= 0.1
            elif speed > 900:
                reward += 0.2
            elif speed > 800:
                reward += 0.1
            elif speed > 600:
                pass  # neutral
            else:
                reward -= 0.1

        # ── Checkpoint progression (reference: getCheckpointReward) ─────
        total_cp = cur_cp + cur_lap * self.TOTAL_CHECKPOINTS

        if self._prev_total_cp is None:
            self._prev_total_cp = total_cp
            self._prev_frame = frame
            self._current_lap = cur_lap
        else:
            if frame < self._prev_frame or frame > self._prev_frame + 60:
                self._prev_total_cp = total_cp
            self._prev_frame = frame

            cp_delta = total_cp - self._prev_total_cp
            # Ignore large discontinuities from resets/teleports to avoid reward spikes.
            if abs(cp_delta) > 3:
                cp_delta = 0
            if cp_delta * 10 >= -5000:
                reward += cp_delta * 10

            if cp_delta > 0:
                self._steps_since_last_cp = 0
            else:
                self._steps_since_last_cp += 1

            self._prev_total_cp = total_cp

            if cur_lap > self._current_lap:
                self._laps_completed += 1
                self._current_lap = cur_lap

        startup_grace = self._episode_steps < self.GRACE_STEPS

        # ── On/off-track shaping (blog-style dense signal) ───────────────
        if surface == self.SURFACE_ROAD:
            reward += 0.1
        elif surface == self.SURFACE_GRASS and not startup_grace:
            reward -= 0.5

        # ── Experimental penalties (reference: getExperimentalReward) ───
        # Apply on transition into a hazard to prevent per-frame punishment spikes.
        hazard_surfaces = {
            self.SURFACE_WALL,
            self.SURFACE_FALL,
            self.SURFACE_DEEP,
            self.SURFACE_WATER,
        }
        in_hazard = surface in hazard_surfaces
        was_in_hazard = self._prev_surface in hazard_surfaces
        if (not startup_grace) and in_hazard and not was_in_hazard:
            reward -= 1.0
        self._prev_surface = surface

        # ── Wall accumulation termination ─────────────────────────────────
        if not startup_grace:
            self._wall_steps += 1
            if surface in (self.SURFACE_WALL, self.SURFACE_FALL, self.SURFACE_DEEP):
                self._wall_hits += 1
            elif surface == self.SURFACE_WATER:
                self._wall_hits += 0.05
            if self._wall_hits >= 5:
                self._wall_hits = 0
                self._wall_steps = 0
                return reward, True
            if self._wall_steps >= 500:
                self._wall_hits = 0
                self._wall_steps = 0

        # ── Other termination ─────────────────────────────────────────────
        if cur_lap >= self.MAX_LAPS:
            return reward, True

        return reward, terminated or truncated

    # ── Observation ───────────────────────────────────────────────────────────
    # Following grumpyoldnerd.com minimap approach:
    #   1. Read 128x128 track tilemap from RAM
    #   2. Physics lookup: tile byte -> surface type -> grayscale value
    #   3. Crop 20x20 around player kart
    #   4. Grayscale background (road=white, grass=gray, wall=black)
    #   5. Overlay player kart in red, enemy karts in blue

    # Enemy kart RAM addresses — stride 0x40 (64 bytes) per kart
    # Kart 0 (player): X=136, Y=140 | Kart 1: X=200, Y=204 | etc.
    ENEMY_KART_OFFSETS = [
        (200, 204),
        (264, 268),
        (328, 332),
        (392, 396),
        (456, 460),
        (520, 524),
        (584, 588),
    ]

    def _get_minimap(self):
        ram = self.env.get_ram()

        # 1. Read 128x128 track tilemap from RAM at 0x10000
        tilemap_offset = 0x10000
        tilemap_size = 128 * 128
        if len(ram) >= tilemap_offset + tilemap_size:
            tilemap = np.frombuffer(
                ram[tilemap_offset : tilemap_offset + tilemap_size], dtype=np.uint8
            ).reshape((128, 128))
        else:
            tilemap = np.zeros((128, 128), dtype=np.uint8)

        # 2. Physics lookup: RAM at 0xB00 + tile_id gives physics byte
        #    0x40=road, 0x42=ghost road, 0x4C=choco road, 0x80=wall,
        #    0x20=pit, 0x22=deep water, 0x54/0x5A=dirt, 0x10=jump, 0x16=boost
        physics_table = (
            ram[0xB00 : 0xB00 + 256]
            if len(ram) >= 0xB00 + 256
            else np.zeros(256, dtype=np.uint8)
        )
        physics_map = np.array(physics_table, dtype=np.uint8)[tilemap]

        # Convert physics values to grayscale
        gray = np.full((128, 128), 100, dtype=np.uint8)  # unknown = mid gray
        gray[physics_map == 0x40] = 200  # road         = white
        gray[physics_map == 0x42] = 200  # ghost road    = white
        gray[physics_map == 0x4C] = 200  # choco road    = white
        gray[physics_map == 0x10] = 220  # jump pad      = bright
        gray[physics_map == 0x16] = 240  # speed boost   = very bright
        gray[physics_map == 0x80] = 0  # wall          = black
        gray[physics_map == 0x20] = 30  # pit           = very dark
        gray[physics_map == 0x22] = 30  # deep water    = very dark
        gray[physics_map == 0x54] = 80  # dirt          = darker gray
        gray[physics_map == 0x5A] = 80  # dirt variant  = darker gray

        # Convert to 3-channel so we can overlay colored kart dots
        img = np.stack([gray, gray, gray], axis=-1)  # (128, 128, 3)

        # Get player kart position and angle
        kart_x = int(np.frombuffer(ram[136:138], dtype=np.dtype("<i2"))[0])
        kart_y = int(np.frombuffer(ram[140:142], dtype=np.dtype("<i2"))[0])
        tx = max(0, min(127, kart_x // 8))
        ty = max(0, min(127, kart_y // 8))

        angle_raw = int(np.frombuffer(ram[149:150], dtype=np.dtype("|u1"))[0])
        angle_deg = (angle_raw / 256.0) * 360.0

        # 5. Overlay enemy karts in blue (before rotation so they rotate with the map)
        for x_off, y_off in self.ENEMY_KART_OFFSETS:
            ex = int(np.frombuffer(ram[x_off : x_off + 2], dtype=np.dtype("<i2"))[0])
            ey = int(np.frombuffer(ram[y_off : y_off + 2], dtype=np.dtype("<i2"))[0])
            etx = max(0, min(127, ex // 8))
            ety = max(0, min(127, ey // 8))
            img[ety, etx] = [0, 0, 255]  # blue

        # Rotate around kart position to keep player always facing "up"
        M = cv2.getRotationMatrix2D((tx, ty), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            img, M, (128, 128), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0)
        )

        # 3. Crop 20x20 centered on kart
        half = self.FRAME_W // 2
        crop = np.zeros((self.FRAME_H, self.FRAME_W, 3), dtype=np.uint8)

        x_min = max(0, tx - half)
        x_max = min(128, tx + half)
        y_min = max(0, ty - half)
        y_max = min(128, ty + half)

        c_x_min = half - (tx - x_min)
        c_x_max = half + (x_max - tx)
        c_y_min = half - (ty - y_min)
        c_y_max = half + (y_max - ty)

        if c_y_max > c_y_min and c_x_max > c_x_min:
            crop[c_y_min:c_y_max, c_x_min:c_x_max] = rotated[y_min:y_max, x_min:x_max]

        # 5. Overlay player kart in red at center
        if half < self.FRAME_H and half < self.FRAME_W:
            crop[half, half] = [255, 0, 0]

        return crop

    def _get_obs(self):
        image = np.concatenate(self._frame_stack, axis=-1)  # (H, W, N_STACK*C)
        image = np.transpose(image, (2, 0, 1))  # (C, H, W) for PyTorch
        # Copy avoids accidental in-place mutation by downstream wrappers/policies.
        return {
            "image": image,
            "features": self._last_features.copy(),
        }

    # ── Gym interface ─────────────────────────────────────────────────────────
    def reset(self, **kwargs):
        obs, info = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self._reset_episode_state()
        # Prime scalar features on reset so the first observation is fully populated.
        _, checkpoint, speed, surface, vx, vy, _, _, _, _, _ = self._read_ram()
        self._last_features = self._make_features(checkpoint, speed, surface, vx, vy)
        frame = self._get_minimap()
        self._frame_stack = [frame] * self.N_STACK
        return self._get_obs(), info

    def step(self, action):
        act = self._actions[action]
        total_reward = 0.0
        done = False
        prev_obs = None

        for i in range(self.FRAME_SKIP):
            obs, _reward, terminated, truncated, info = self.env.step(act)
            self._episode_steps += 1

            if self.render_mode == "human":
                img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (512, 448), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Super Mario Kart AI", img)
                cv2.waitKey(1)

            (
                lap,
                checkpoint,
                speed,
                surface,
                vx,
                vy,
                game_mode,
                frame,
                curr_min,
                curr_sec,
                turned,
            ) = self._read_ram()
            # Features are updated every emulator step before the final stacked obs is returned.
            self._last_features = self._make_features(
                checkpoint, speed, surface, vx, vy
            )

            step_reward, step_done = self._compute_reward_and_done(
                lap,
                checkpoint,
                speed,
                surface,
                terminated,
                truncated,
                game_mode,
                frame,
                curr_min,
                curr_sec,
                turned,
            )
            total_reward += step_reward

            # Keep last 2 frames for MaxAndSkip
            if i >= self.FRAME_SKIP - 2:
                prev_obs = obs if prev_obs is None else prev_obs
            if step_done:
                done = True
                break

        # Max of last 2 frames — prevents flickering (matches reference MaxAndSkipEnv)
        if prev_obs is not None and not done:
            obs = np.maximum(prev_obs, obs)

        self._frame_stack.pop(0)
        self._frame_stack.append(self._get_minimap())

        # Clip to [-1, +1] — preserves fractional magnitudes from blog reward
        clipped_reward = float(np.clip(total_reward, -1.0, 1.0))
        clipped_reward *= self.reward_scale

        return self._get_obs(), clipped_reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.render()

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
        self.env.close()


def make_env(
    game,
    state,
    render_mode=None,
    grayscale=False,
    speed_reward=False,
    reward_scale=1.0,
):
    return RetroToGym(
        game,
        state=state,
        render_mode=render_mode,
        grayscale=grayscale,
        speed_reward=speed_reward,
        reward_scale=reward_scale,
    )
