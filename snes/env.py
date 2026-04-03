"""
Game environment wrapper for Super Mario Kart (SNES).
Handles frame stacking, action discretisation, reward shaping, and early termination.
RAM addresses verified by scan_ram.py and test_surface2.py.

Observation is a Dict with:
  "pixels"   — (84, 84, N_STACK) uint8 stacked grayscale frames
  "features" — (N_FEATURES,) float32: [direction_norm, checkpoint_norm, speed_norm]
"""

import retro
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2


class RetroToGym(gym.Env):

    # Race constants (confirmed by RAM scan)
    LAP_BASE          = 127  # raw lap byte at race start
    TOTAL_CHECKPOINTS = 30   # confirmed by totalCheckpoints RAM address
    MAX_LAPS          = 1    # number of laps to complete from start of episode

    # Episode limits
    GRACE_STEPS           = 0      # no countdown in this savestate
    NO_PROGRESS_STEPS     = 500    # frames since last checkpoint — wall-hugger dies in ~8 sec

    FRAME_SKIP         = 4    # repeat each action for this many frames
    N_STACK            = 4    # frames to stack for temporal context
    N_CHANNELS_PER_FRAME = 3  # default; overridden to 1 per frame when grayscale=True
    N_ACTIONS          = 4    # straight, left, right, brake
    N_FEATURES         = 5    # [checkpoint_norm, speed_norm, surface_norm, vx_norm, vy_norm]
    FRAME_W            = 96   # width after resize
    FRAME_H            = 72   # height after resize (crop then scale)
    CROP_BOTTOM        = 112  # crop bottom half (minimap) before resize

    # Surface values confirmed by test_surface2.py
    SURFACE_ROAD  = 64   # clean road
    SURFACE_GRASS = 84   # off-track grass
    SURFACE_WALL  = 128  # wall contact (1 frame per bounce)
    # Confirmed by test_rbroad_surface.py — Rainbow Road void/fall value.
    # Appears when kart falls off the edge; speed drops to 0 until Lakitu rescues.
    SURFACE_FALL  = 40

    def __init__(self, game, state=retro.State.NONE, render_mode=None, grayscale=False):
        self.env = retro.make(game=game, state=state, render_mode="rgb_array")
        self.grayscale = grayscale
        self.N_CHANNELS_PER_FRAME = 1 if grayscale else 3
        self._actions     = self._build_actions()
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(self.FRAME_H, self.FRAME_W, self.N_STACK * self.N_CHANNELS_PER_FRAME),
                dtype=np.uint8
            ),
            "features": spaces.Box(
                low=-1.0, high=1.0, shape=(self.N_FEATURES,), dtype=np.float32
            ),
        })
        self._frame_stack = []
        self.render_mode  = render_mode
        self._reset_episode_state()

    # ── Action set ────────────────────────────────────────────────────────────
    def _build_actions(self):
        # Button order: B Y SELECT START UP DOWN LEFT RIGHT A X L R
        combos = [
            [0],      # 0  B         — accelerate straight
            [0, 6],   # 1  B + LEFT  — accelerate + steer left
            [0, 7],   # 2  B + RIGHT — accelerate + steer right
            [1],      # 3  Y         — brake
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
        ram     = self.env.get_ram()
        lap     = int(np.frombuffer(ram[4289:4290], dtype=np.dtype("|u1"))[0])
        cp      = int(np.frombuffer(ram[4316:4317], dtype=np.dtype("|u1"))[0])
        speed   = int(np.frombuffer(ram[4330:4332], dtype=np.dtype("<i2"))[0])
        surface = int(np.frombuffer(ram[4270:4271], dtype=np.dtype("|u1"))[0])
        vx      = int(np.frombuffer(ram[4130:4132], dtype=np.dtype("<i2"))[0])  # 0x1022 X velocity
        vy      = int(np.frombuffer(ram[4132:4134], dtype=np.dtype("<i2"))[0])  # 0x1024 Y velocity
        return lap, cp, speed, surface, vx, vy

    def _make_features(self, cp, speed, surface, vx, vy):
        # surface encoded as 0=road, 0.5=grass/fall, 1.0=wall
        surface_norm = {
            self.SURFACE_ROAD:  0.0,
            self.SURFACE_GRASS: 0.5,
            self.SURFACE_WALL:  1.0,
            self.SURFACE_FALL:  0.5,
        }.get(surface, 0.5)
        # velocity components normalised to [-1, 1] — max observed ~600
        vx_norm = max(-1.0, min(int(vx) / 600.0, 1.0))
        vy_norm = max(-1.0, min(int(vy) / 600.0, 1.0))
        return np.array([
            cp / float(self.TOTAL_CHECKPOINTS - 1),
            max(0, min(int(speed), 1000)) / 1000.0,
            surface_norm,
            vx_norm,
            vy_norm,
        ], dtype=np.float32)

    def _reset_episode_state(self):
        self._episode_steps      = 0
        self._current_lap        = None
        self._start_lap          = None   # lap value at episode start
        self._laps_completed     = 0
        self._earned_cps         = set()
        self._steps_since_last_cp = 0
        self._last_features      = np.zeros(self.N_FEATURES, dtype=np.float32)

    # ── Reward & termination ──────────────────────────────────────────────────
    def _compute_reward_and_done(self, lap, checkpoint, speed, surface, terminated, truncated):
        reward   = 0.0
        in_grace = self._episode_steps < self.GRACE_STEPS

        cur_lap = int(lap) - self.LAP_BASE
        cur_cp  = int(checkpoint)

        # ── Dense speed reward only on road — no reward for wall/grass speed ──
        speed_norm = min(max(int(speed), 0), 1000) / 1000.0
        if surface == self.SURFACE_ROAD:
            reward += speed_norm * 0.05   # max +0.05/frame on road
        reward -= 0.005  # survival penalty

        if self._current_lap is None:
            self._current_lap = cur_lap
            self._start_lap   = cur_lap
            self._earned_cps  = {cur_cp}
            self._steps_since_last_cp = 0
        else:
            if cur_lap > self._current_lap:
                self._laps_completed += 1
                reward += 10.0   # lap bonus
                self._current_lap = cur_lap
                self._earned_cps  = set()
                self._steps_since_last_cp = 0

            if cur_cp not in self._earned_cps:
                self._earned_cps.add(cur_cp)
                reward += 2.0 * speed_norm  # base: speed-weighted
                if surface == self.SURFACE_ROAD:
                    reward += 5.0  # road checkpoint bonus
                self._steps_since_last_cp = 0
            else:
                self._steps_since_last_cp += 1

        if not in_grace:
            if surface == self.SURFACE_FALL:
                reward -= 0.5
            elif surface == self.SURFACE_GRASS:
                reward -= 0.03
            elif surface == self.SURFACE_WALL:
                reward -= 0.03

        if self._laps_completed >= self.MAX_LAPS:
            return reward, True

        no_progress = self._steps_since_last_cp >= self.NO_PROGRESS_STEPS
        return reward, terminated or truncated or no_progress

    # ── Observation ───────────────────────────────────────────────────────────
    def _preprocess_frame(self, obs):
        cropped = obs[:self.CROP_BOTTOM, :]
        frame = cv2.resize(cropped, (self.FRAME_W, self.FRAME_H), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]  # (H, W, 1)
        return frame

    def _get_obs(self):
        return {
            "pixels":   np.concatenate(self._frame_stack, axis=-1),  # (H, W, N_STACK*3)
            "features": self._last_features.copy(),
        }

    # ── Gym interface ─────────────────────────────────────────────────────────
    def reset(self, **kwargs):
        obs, info = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self._reset_episode_state()
        frame = self._preprocess_frame(obs)
        self._frame_stack = [frame] * self.N_STACK
        return self._get_obs(), info

    def step(self, action):
        act = self._actions[action]
        total_reward = 0.0
        done = False

        for _ in range(self.FRAME_SKIP):
            obs, _reward, terminated, truncated, info = self.env.step(act)
            self._episode_steps += 1

            if self.render_mode == "human":
                img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (512, 448), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Super Mario Kart AI", img)
                cv2.waitKey(1)

            lap, checkpoint, speed, surface, vx, vy = self._read_ram()
            self._last_features = self._make_features(checkpoint, speed, surface, vx, vy)

            step_reward, step_done = self._compute_reward_and_done(
                lap, checkpoint, speed, surface, terminated, truncated
            )
            total_reward += step_reward
            if step_done:
                done = True
                break

        self._frame_stack.pop(0)
        self._frame_stack.append(self._preprocess_frame(obs))

        return self._get_obs(), total_reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.render()

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
        self.env.close()


def make_env(game, state, render_mode=None, grayscale=False):
    return RetroToGym(game, state=state, render_mode=render_mode, grayscale=grayscale)
