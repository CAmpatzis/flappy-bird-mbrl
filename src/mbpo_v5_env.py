"""Headless feature-only Flappy Bird environment for MBPO v5.

The RL loop no longer depends on pygame rendering. The environment keeps only
the compact feature state used by the policy and world model:

    [bird_y, bird_vel, pipe1_dx, pipe1_gap_y, pipe2_dx, pipe2_gap_y, ...]

This makes both real rollouts and model learning operate on the same 8D state
representation, which removes the replay-buffer shape mismatch that affected the
earlier image-based MBPO variants.
"""

from __future__ import annotations

import numpy as np

SCREEN_W = 288.0
SCREEN_H = 512.0
BASE_Y = SCREEN_H * 0.79

PIPE_GAP = 100.0
PIPE_W = 52.0
PIPE_H = 320.0
PIPE_SPEED = -4.0
NEW_PIPE_X = SCREEN_W + 10.0

PLAYER_X = int(SCREEN_W * 0.2)
PLAYER_W = 34.0
PLAYER_H = 24.0
INITIAL_PLAYER_Y = (SCREEN_H - PLAYER_H) / 2.0

PLAYER_MAX_VEL_Y = 10.0
PLAYER_ACC_Y = 1.0
PLAYER_FLAP_ACC = -9.0

NUM_FEATURES = 8
DEFAULT_GAP_CENTER = SCREEN_H / 2.0
INVALID_PIPE_DX = 1.5

_GAP_TOP_CHOICES = np.array([20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float32)
_GAP_TOP_CHOICES += int(BASE_Y * 0.2)
GAP_CENTER_CHOICES = _GAP_TOP_CHOICES + PIPE_GAP / 2.0


def feature_potential(features: np.ndarray) -> np.ndarray:
    """Potential used for dense PBRS shaping on the upcoming pipe gap."""
    bird_y = features[..., 0]
    next_gap_y = features[..., 3]
    return -np.abs(bird_y - next_gap_y)


class HeadlessFlappyBirdVecEnv:
    """Vectorized Flappy Bird dynamics without rendering.

    The environment tracks the same physics constants as the original pygame
    implementation, but it exposes only the normalized 8D feature vector.
    """

    def __init__(self, num_envs: int, seed: int | None = None, max_episode_steps: int | None = None):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.num_envs = int(num_envs)
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        self.player_y = np.zeros(self.num_envs, dtype=np.float32)
        self.player_vel = np.zeros(self.num_envs, dtype=np.float32)
        self.pipe_x = np.full((self.num_envs, 3), np.inf, dtype=np.float32)
        self.gap_center = np.full((self.num_envs, 3), DEFAULT_GAP_CENTER, dtype=np.float32)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.reset()

    def _sample_gap_centers(self, size: int) -> np.ndarray:
        indices = self.rng.integers(0, len(GAP_CENTER_CHOICES), size=size)
        return GAP_CENTER_CHOICES[indices].astype(np.float32, copy=False)

    def _reset_indices(self, indices: np.ndarray) -> None:
        count = len(indices)
        if count == 0:
            return

        self.player_y[indices] = INITIAL_PLAYER_Y
        self.player_vel[indices] = 0.0
        self.pipe_x[indices] = np.inf
        self.gap_center[indices] = DEFAULT_GAP_CENTER

        self.pipe_x[indices, 0] = SCREEN_W
        self.pipe_x[indices, 1] = SCREEN_W + SCREEN_W / 2.0
        self.gap_center[indices, 0] = self._sample_gap_centers(count)
        self.gap_center[indices, 1] = self._sample_gap_centers(count)
        self.episode_steps[indices] = 0

    def reset(self, mask: np.ndarray | list[int] | None = None) -> np.ndarray:
        if mask is None:
            indices = np.arange(self.num_envs)
        else:
            mask_array = np.asarray(mask)
            if mask_array.dtype == np.bool_:
                if mask_array.shape != (self.num_envs,):
                    raise ValueError("Boolean reset mask must have shape (num_envs,)")
                indices = np.flatnonzero(mask_array)
            else:
                indices = mask_array.astype(np.int64, copy=False).reshape(-1)

        self._reset_indices(indices)
        return self.features()[indices].copy()

    def features(self) -> np.ndarray:
        features = np.zeros((self.num_envs, NUM_FEATURES), dtype=np.float32)
        features[:, 0] = self.player_y / SCREEN_H
        features[:, 1] = self.player_vel / 10.0

        valid = np.isfinite(self.pipe_x)
        dx = np.where(valid, (self.pipe_x - PLAYER_X) / SCREEN_W, INVALID_PIPE_DX)
        gap = np.where(valid, self.gap_center / SCREEN_H, DEFAULT_GAP_CENTER / SCREEN_H)

        features[:, 2] = dx[:, 0]
        features[:, 3] = gap[:, 0]
        features[:, 4] = dx[:, 1]
        features[:, 5] = gap[:, 1]
        features[:, 6] = dx[:, 2]
        features[:, 7] = gap[:, 2]
        return features

    def _spawn_new_pipe(self) -> None:
        spawn_mask = (
            (self.pipe_x[:, 0] > 0.0)
            & (self.pipe_x[:, 0] < abs(PIPE_SPEED) + 1.0)
            & (~np.isfinite(self.pipe_x[:, 2]))
        )
        if not np.any(spawn_mask):
            return

        count = int(spawn_mask.sum())
        self.pipe_x[spawn_mask, 2] = NEW_PIPE_X
        self.gap_center[spawn_mask, 2] = self._sample_gap_centers(count)

    def _remove_passed_pipe(self) -> None:
        remove_mask = self.pipe_x[:, 0] < -PIPE_W
        if not np.any(remove_mask):
            return

        old_pipe_x = self.pipe_x[remove_mask].copy()
        old_gap_center = self.gap_center[remove_mask].copy()

        self.pipe_x[remove_mask, 0] = old_pipe_x[:, 1]
        self.pipe_x[remove_mask, 1] = old_pipe_x[:, 2]
        self.pipe_x[remove_mask, 2] = np.inf

        self.gap_center[remove_mask, 0] = old_gap_center[:, 1]
        self.gap_center[remove_mask, 1] = old_gap_center[:, 2]
        self.gap_center[remove_mask, 2] = DEFAULT_GAP_CENTER

    def _pipe_collision(self) -> np.ndarray:
        bird_top = self.player_y[:, None]
        bird_bottom = bird_top + PLAYER_H

        pipe_left = self.pipe_x
        pipe_right = self.pipe_x + PIPE_W
        x_overlap = (pipe_left < PLAYER_X + PLAYER_W) & (pipe_right > PLAYER_X)

        gap_top = self.gap_center - PIPE_GAP / 2.0
        gap_bottom = self.gap_center + PIPE_GAP / 2.0
        outside_gap = (bird_top < gap_top) | (bird_bottom > gap_bottom)

        return np.any(x_overlap & outside_gap & np.isfinite(self.pipe_x), axis=1)

    def step(self, actions: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action_array = np.asarray(actions, dtype=np.int64).reshape(-1)
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"Expected actions with shape ({self.num_envs},), got {action_array.shape}")
        if np.any((action_array != 0) & (action_array != 1)):
            raise ValueError("Actions must be 0 (no-op) or 1 (flap)")

        raw_rewards = np.zeros(self.num_envs, dtype=np.float32)

        flap_mask = (action_array == 1) & (self.player_y > -2.0 * PLAYER_H)
        self.player_vel[flap_mask] = PLAYER_FLAP_ACC

        player_mid = PLAYER_X + PLAYER_W / 2.0
        pipe_mid = self.pipe_x + PIPE_W / 2.0
        pass_mask = np.any((pipe_mid <= player_mid) & (player_mid < pipe_mid + abs(PIPE_SPEED)), axis=1)
        raw_rewards[pass_mask] = 1.0

        gravity_mask = (~flap_mask) & (self.player_vel < PLAYER_MAX_VEL_Y)
        self.player_vel[gravity_mask] += PLAYER_ACC_Y

        max_drop = BASE_Y - self.player_y - PLAYER_H
        self.player_y += np.minimum(self.player_vel, max_drop)
        self.player_y = np.maximum(self.player_y, 0.0)

        finite_pipe_mask = np.isfinite(self.pipe_x)
        self.pipe_x[finite_pipe_mask] += PIPE_SPEED
        self.episode_steps += 1

        self._spawn_new_pipe()
        self._remove_passed_pipe()

        done = self._pipe_collision() | (self.player_y + PLAYER_H >= BASE_Y - 1.0)
        if self.max_episode_steps is not None:
            done |= self.episode_steps >= self.max_episode_steps

        raw_rewards[done] = -1.0
        return self.features().copy(), raw_rewards, done.astype(np.bool_)