"""MBPO v6 implemented in JAX.

Design goals:
- Preserve the feature-only MBPO formulation introduced in v5.
- Move model learning, policy updates, and rollout generation onto JAX/Optax.
- Prefer Apple Metal when the JAX Metal plugin is present, with a CPU fallback.
- Use an aggressive fast-learning schedule because the 8D Flappy Bird dynamics
    are simple enough to support heavier MBPO exploitation than the v5 defaults.

The implementation is tuned around JIT-compiled policy updates, ensemble world
model training, and short model rollouts. On Apple Silicon, Metal is used when
available; CPU thread limits are still applied to keep host-side work bounded.
"""

from __future__ import annotations

import json
import importlib.util
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time as time_module
from functools import partial
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple


def _recommended_cpu_threads(fraction: float = 0.5) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, int(round(cpu_count * fraction))))


def _default_jax_platform() -> str:
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("jax_plugins.metal_plugin") is not None
    ):
        return "METAL"
    return "cpu"


def _normalize_jax_platform_name(name: str) -> str:
    normalized = name.strip()
    if normalized.lower() == "metal":
        return "METAL"
    return normalized.lower()


def _probe_metal_backend() -> tuple[bool, str | None]:
    probe_code = "\n".join(
        [
            "import os",
            "os.environ.setdefault('ENABLE_PJRT_COMPATIBILITY', '1')",
            "os.environ.pop('JAX_PLATFORM_NAME', None)",
            "import jax",
            "import jax.numpy as jnp",
            "backend = str(jax.default_backend()).lower()",
            "if backend != 'metal':",
            "    raise RuntimeError(f'backend={backend}')",
            "jnp.arange(4, dtype=jnp.float32).block_until_ready()",
            "jax.random.PRNGKey(0)",
            "print('metal-ok')",
        ]
    )
    probe_env = os.environ.copy()
    probe_env.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
    probe_env.pop("JAX_PLATFORM_NAME", None)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            env=probe_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    if completed.returncode == 0:
        return True, None

    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    failure_candidates = []
    for line in (stderr + "\n" + stdout).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        failure_candidates.append(stripped)

    failure_line = None
    for token in ("default_memory_space", "UNIMPLEMENTED", "RuntimeError", "backend="):
        match = next((line for line in failure_candidates if token in line), None)
        if match is not None:
            failure_line = match
            break

    if failure_line is None:
        failure = stderr or stdout or f"probe failed with code {completed.returncode}"
        failure_line = failure.splitlines()[-1]

    failure_line = failure_line[:240]
    return False, failure_line


def _configure_default_runtime() -> int:
    cpu_fraction = float(os.environ.get("MBPO_V6_CPU_FRACTION", "0.5"))
    cpu_threads = _recommended_cpu_threads(cpu_fraction)

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", os.environ.get("MBPO_V6_MEMORY_FRACTION", "0.5"))
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "intra_op_parallelism_threads" not in xla_flags:
        extra = (
            " --xla_cpu_multi_thread_eigen=true"
            f" intra_op_parallelism_threads={cpu_threads}"
        )
        os.environ["XLA_FLAGS"] = (xla_flags + extra).strip()

    preferred_platform = _normalize_jax_platform_name(
        os.environ.get("MBPO_V6_PLATFORM", _default_jax_platform())
    )
    if preferred_platform == "METAL":
        force_metal = os.environ.get("MBPO_V6_FORCE_METAL", "0") == "1"
        if force_metal:
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["MBPO_V6_EFFECTIVE_PLATFORM"] = "METAL"
            os.environ.pop("MBPO_V6_PLATFORM_FALLBACK_REASON", None)
        else:
            metal_ok, failure = _probe_metal_backend()
            if metal_ok:
                os.environ.pop("JAX_PLATFORM_NAME", None)
                os.environ["MBPO_V6_EFFECTIVE_PLATFORM"] = "METAL"
                os.environ.pop("MBPO_V6_PLATFORM_FALLBACK_REASON", None)
            else:
                os.environ["JAX_PLATFORM_NAME"] = "cpu"
                os.environ["MBPO_V6_EFFECTIVE_PLATFORM"] = "cpu"
                os.environ["MBPO_V6_PLATFORM_FALLBACK_REASON"] = failure or "Metal probe failed"
    else:
        os.environ.setdefault("JAX_PLATFORM_NAME", preferred_platform)
        os.environ["MBPO_V6_EFFECTIVE_PLATFORM"] = preferred_platform
        os.environ.pop("MBPO_V6_PLATFORM_FALLBACK_REASON", None)
    return cpu_threads


DEFAULT_CPU_THREADS = _configure_default_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", False)


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

GAP_CENTER_CHOICES = np.asarray([20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float32)
GAP_CENTER_OFFSET = float(int(BASE_Y * 0.2) + PIPE_GAP / 2.0)


class EnvState(NamedTuple):
    player_y: jnp.ndarray
    player_vel: jnp.ndarray
    pipe_x: jnp.ndarray
    gap_center: jnp.ndarray
    episode_steps: jnp.ndarray


def feature_potential(features: jnp.ndarray) -> jnp.ndarray:
    return -jnp.abs(features[..., 0] - features[..., 3])


def clamp_features(features: jnp.ndarray) -> jnp.ndarray:
    clamped = features
    clamped = clamped.at[..., 0].set(jnp.clip(clamped[..., 0], -0.20, 1.20))
    clamped = clamped.at[..., 1].set(jnp.clip(clamped[..., 1], -1.20, 1.20))
    clamped = clamped.at[..., 2::2].set(jnp.clip(clamped[..., 2::2], -0.50, 2.00))
    clamped = clamped.at[..., 3::2].set(jnp.clip(clamped[..., 3::2], 0.0, 1.0))
    return clamped


@jax.jit
def extract_features(env_state: EnvState) -> jnp.ndarray:
    valid = jnp.isfinite(env_state.pipe_x)
    dx = jnp.where(valid, (env_state.pipe_x - PLAYER_X) / SCREEN_W, INVALID_PIPE_DX)
    gap = jnp.where(valid, env_state.gap_center / SCREEN_H, DEFAULT_GAP_CENTER / SCREEN_H)

    return jnp.stack(
        [
            env_state.player_y / SCREEN_H,
            env_state.player_vel / 10.0,
            dx[:, 0],
            gap[:, 0],
            dx[:, 1],
            gap[:, 1],
            dx[:, 2],
            gap[:, 2],
        ],
        axis=-1,
    )


def _sample_gap_centers(key: jax.Array, count: int) -> jnp.ndarray:
    indices = jax.random.randint(key, (count,), 0, GAP_CENTER_CHOICES.shape[0])
    return jnp.take(jnp.asarray(GAP_CENTER_CHOICES) + GAP_CENTER_OFFSET, indices, axis=0)


def reset_env(key: jax.Array, num_envs: int) -> tuple[EnvState, jnp.ndarray]:
    key_1, key_2 = jax.random.split(key)
    first_gap = _sample_gap_centers(key_1, num_envs)
    second_gap = _sample_gap_centers(key_2, num_envs)

    pipe_x = jnp.full((num_envs, 3), jnp.inf, dtype=jnp.float32)
    gap_center = jnp.full((num_envs, 3), DEFAULT_GAP_CENTER, dtype=jnp.float32)

    pipe_x = pipe_x.at[:, 0].set(SCREEN_W)
    pipe_x = pipe_x.at[:, 1].set(SCREEN_W + SCREEN_W / 2.0)
    gap_center = gap_center.at[:, 0].set(first_gap)
    gap_center = gap_center.at[:, 1].set(second_gap)

    env_state = EnvState(
        player_y=jnp.full((num_envs,), INITIAL_PLAYER_Y, dtype=jnp.float32),
        player_vel=jnp.zeros((num_envs,), dtype=jnp.float32),
        pipe_x=pipe_x,
        gap_center=gap_center,
        episode_steps=jnp.zeros((num_envs,), dtype=jnp.int32),
    )
    return env_state, extract_features(env_state)


@jax.jit
def reset_done_envs(key: jax.Array, env_state: EnvState, done_mask: jnp.ndarray) -> tuple[EnvState, jnp.ndarray]:
    num_envs = env_state.player_y.shape[0]
    new_state, _ = reset_env(key, num_envs)

    merged_state = EnvState(
        player_y=jnp.where(done_mask, new_state.player_y, env_state.player_y),
        player_vel=jnp.where(done_mask, new_state.player_vel, env_state.player_vel),
        pipe_x=jnp.where(done_mask[:, None], new_state.pipe_x, env_state.pipe_x),
        gap_center=jnp.where(done_mask[:, None], new_state.gap_center, env_state.gap_center),
        episode_steps=jnp.where(done_mask, new_state.episode_steps, env_state.episode_steps),
    )
    return merged_state, extract_features(merged_state)


@jax.jit
def env_step(
    key: jax.Array,
    env_state: EnvState,
    actions: jnp.ndarray,
    max_episode_steps: int,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    raw_rewards = jnp.zeros((env_state.player_y.shape[0],), dtype=jnp.float32)

    flap_mask = (actions == 1) & (env_state.player_y > -2.0 * PLAYER_H)
    player_vel = jnp.where(flap_mask, PLAYER_FLAP_ACC, env_state.player_vel)

    player_mid = PLAYER_X + PLAYER_W / 2.0
    pipe_mid = env_state.pipe_x + PIPE_W / 2.0
    pass_mask = jnp.any(
        (pipe_mid <= player_mid) & (player_mid < pipe_mid + abs(PIPE_SPEED)),
        axis=1,
    )
    raw_rewards = jnp.where(pass_mask, 1.0, raw_rewards)

    gravity_mask = (~flap_mask) & (player_vel < PLAYER_MAX_VEL_Y)
    player_vel = jnp.where(gravity_mask, player_vel + PLAYER_ACC_Y, player_vel)
    max_drop = BASE_Y - env_state.player_y - PLAYER_H
    player_y = env_state.player_y + jnp.minimum(player_vel, max_drop)
    player_y = jnp.maximum(player_y, 0.0)

    finite_mask = jnp.isfinite(env_state.pipe_x)
    pipe_x = jnp.where(finite_mask, env_state.pipe_x + PIPE_SPEED, env_state.pipe_x)
    gap_center = env_state.gap_center
    episode_steps = env_state.episode_steps + 1

    spawn_mask = (
        (pipe_x[:, 0] > 0.0)
        & (pipe_x[:, 0] < abs(PIPE_SPEED) + 1.0)
        & (~jnp.isfinite(pipe_x[:, 2]))
    )
    new_gap_centers = _sample_gap_centers(key, env_state.player_y.shape[0])
    pipe_x = pipe_x.at[:, 2].set(jnp.where(spawn_mask, NEW_PIPE_X, pipe_x[:, 2]))
    gap_center = gap_center.at[:, 2].set(jnp.where(spawn_mask, new_gap_centers, gap_center[:, 2]))

    remove_mask = pipe_x[:, 0] < -PIPE_W
    shifted_pipe_x = jnp.concatenate([pipe_x[:, 1:], jnp.full((pipe_x.shape[0], 1), jnp.inf, dtype=jnp.float32)], axis=1)
    shifted_gap_center = jnp.concatenate(
        [gap_center[:, 1:], jnp.full((gap_center.shape[0], 1), DEFAULT_GAP_CENTER, dtype=jnp.float32)],
        axis=1,
    )
    pipe_x = jnp.where(remove_mask[:, None], shifted_pipe_x, pipe_x)
    gap_center = jnp.where(remove_mask[:, None], shifted_gap_center, gap_center)

    bird_top = player_y[:, None]
    bird_bottom = bird_top + PLAYER_H
    pipe_left = pipe_x
    pipe_right = pipe_x + PIPE_W
    x_overlap = (pipe_left < PLAYER_X + PLAYER_W) & (pipe_right > PLAYER_X)
    gap_top = gap_center - PIPE_GAP / 2.0
    gap_bottom = gap_center + PIPE_GAP / 2.0
    outside_gap = (bird_top < gap_top) | (bird_bottom > gap_bottom)
    pipe_collision = jnp.any(x_overlap & outside_gap & jnp.isfinite(pipe_x), axis=1)
    ground_collision = player_y + PLAYER_H >= BASE_Y - 1.0
    timeout_collision = (max_episode_steps > 0) & (episode_steps >= max_episode_steps)
    done = pipe_collision | ground_collision | timeout_collision

    raw_rewards = jnp.where(done, -1.0, raw_rewards)
    next_state = EnvState(
        player_y=player_y,
        player_vel=player_vel,
        pipe_x=pipe_x,
        gap_center=gap_center,
        episode_steps=episode_steps,
    )
    return next_state, extract_features(next_state), raw_rewards, done


def shape_rewards(
    states: np.ndarray,
    next_states: np.ndarray,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    alignment_scale: float,
) -> np.ndarray:
    phi_state = -np.abs(states[..., 0] - states[..., 3])
    phi_next = -np.abs(next_states[..., 0] - next_states[..., 3])
    phi_next = np.where(dones, 0.0, phi_next)
    return (raw_rewards + alignment_scale * (gamma * phi_next - phi_state)).astype(np.float32, copy=False)


@jax.jit
def analytic_termination_from_features(features: jnp.ndarray) -> jnp.ndarray:
    bird_top = features[..., 0] * SCREEN_H
    bird_bottom = bird_top + PLAYER_H
    gap_center = features[..., 3] * SCREEN_H
    pipe_dx = features[..., 2] * SCREEN_W

    gap_top = gap_center - PIPE_GAP / 2.0
    gap_bottom = gap_center + PIPE_GAP / 2.0
    horizontal_overlap = (pipe_dx < PLAYER_W) & (pipe_dx > -PIPE_W)
    outside_gap = (bird_top < gap_top) | (bird_bottom > gap_bottom)
    ground_collision = bird_bottom >= BASE_Y - 1.0
    return ground_collision | (horizontal_overlap & outside_gap)


@dataclass(slots=True)
class MBPOv6JAXConfig:
    num_episodes: int = 20_000
    num_envs: int = 96
    seed: int = 7
    max_episode_steps: int = 0

    cpu_fraction: float = 0.5
    cpu_threads: int = DEFAULT_CPU_THREADS
    jax_platform: str = os.environ.get("JAX_PLATFORM_NAME", "cpu")

    real_buffer_size: int = 1_000_000
    model_buffer_size: int = 500_000
    batch_size: int = 512

    gamma: float = 0.99
    sac_lr: float = 3e-4
    model_lr: float = 1e-3
    tau: float = 0.005
    alpha_min: float = 0.01
    target_entropy_ratio: float = 0.9

    policy_hidden_dim: int = 256
    q_hidden_dim: int = 256
    model_hidden_dim: int = 256

    start_model_training_after: int = 4_096
    start_policy_training_after: int = 4_096
    gradient_updates_per_env_step: float = 0.5
    real_fraction: float = 0.05

    ensemble_size: int = 7
    elite_size: int = 5
    holdout_ratio: float = 0.2
    model_batch_size: int = 512
    model_max_epochs: int = 20
    model_patience: int = 4
    model_train_freq: int = 512
    model_train_max_samples: int = 50_000

    rollout_min_length: int = 1
    rollout_max_length: int = 8
    rollout_warmup_steps: int = 5_000
    rollout_loss_ceiling: float = 2e-2
    rollout_loss_floor: float = 2e-3
    imagined_steps_per_real_step: int = 160
    max_rollout_starts: int = 8_192

    alignment_scale: float = 2.0
    death_penalty: float = -1.0

    tensorboard_root: str = "runs"
    checkpoint_dir: str = "checkpoints/mbpo_v6_jax"
    checkpoint_interval_episodes: int | None = 5_000
    log_interval_episodes: int = 100
    run_name: str | None = None


@dataclass
class MBPOv6JAXResult:
    config: MBPOv6JAXConfig
    run_name: str
    backend: str
    cpu_threads: int
    policy_params: Any
    q1_params: Any
    q2_params: Any
    world_model_params: Any
    elite_indices: list[int]
    raw_rewards: list[float] = field(default_factory=list)
    shaped_rewards: list[float] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    training_time_s: float = 0.0
    best_raw_reward: float = float("-inf")
    best_score: int = 0
    final_model_metrics: dict[str, Any] | None = None


def make_half_power_config(num_episodes: int = 20_000, cpu_fraction: float = 0.5) -> MBPOv6JAXConfig:
    cpu_threads = _recommended_cpu_threads(cpu_fraction)
    backend = str(jax.default_backend()).lower()
    use_metal = backend == "metal"
    return MBPOv6JAXConfig(
        num_episodes=num_episodes,
        num_envs=128 if use_metal else cpu_threads * 16,
        cpu_fraction=cpu_fraction,
        cpu_threads=cpu_threads,
        jax_platform=backend,
        batch_size=768 if use_metal else (512 if cpu_threads >= 6 else 256),
        start_model_training_after=1024,
        start_policy_training_after=1024,
        gradient_updates_per_env_step=0.75 if use_metal else (0.5 if cpu_threads >= 6 else 0.25),
        real_fraction=0.04 if use_metal else 0.05,
        model_train_freq=256,
        model_max_epochs=15,
        model_patience=3,
        rollout_max_length=10,
        rollout_warmup_steps=2048,
        imagined_steps_per_real_step=256 if use_metal else (160 if cpu_threads >= 6 else 96),
        max_rollout_starts=4096,
    )


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int = NUM_FEATURES):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        batch_size = states.shape[0]
        if batch_size == 0:
            return
        if states.shape != next_states.shape:
            raise ValueError(f"next_states shape {next_states.shape} does not match states {states.shape}")
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must have shape [B, {self.state_dim}], got {states.shape}")
        if not (actions.shape[0] == rewards.shape[0] == dones.shape[0] == batch_size):
            raise ValueError("ReplayBuffer batch fields do not all have the same length")

        indices = (np.arange(batch_size) + self.ptr) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> tuple[jnp.ndarray, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            jnp.asarray(self.states[indices]),
            jnp.asarray(self.actions[indices]),
            jnp.asarray(self.rewards[indices]),
            jnp.asarray(self.next_states[indices]),
            jnp.asarray(self.dones[indices]),
        )

    def sample_states(self, count: int) -> jnp.ndarray:
        indices = np.random.randint(0, self.size, size=count)
        return jnp.asarray(self.states[indices])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _init_layer(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jnp.ndarray]:
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    weights = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": weights, "b": bias}


def init_mlp(key: jax.Array, layer_dims: list[int]) -> list[dict[str, jnp.ndarray]]:
    keys = jax.random.split(key, len(layer_dims) - 1)
    return [
        _init_layer(layer_key, in_dim, out_dim)
        for layer_key, in_dim, out_dim in zip(keys, layer_dims[:-1], layer_dims[1:])
    ]


def apply_mlp(params: list[dict[str, jnp.ndarray]], inputs: jnp.ndarray, activation) -> jnp.ndarray:
    outputs = inputs
    for layer in params[:-1]:
        outputs = activation(outputs @ layer["w"] + layer["b"])
    return outputs @ params[-1]["w"] + params[-1]["b"]


def stack_ensemble_params(param_list: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: jnp.stack(items, axis=0), *param_list)


def init_ensemble_mlp(key: jax.Array, ensemble_size: int, layer_dims: list[int]) -> Any:
    keys = jax.random.split(key, ensemble_size)
    return stack_ensemble_params([init_mlp(model_key, layer_dims) for model_key in keys])


def select_ensemble_members(model_params: Any, indices: list[int] | np.ndarray | jnp.ndarray) -> Any:
    index_array = jnp.asarray(indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda values: values[index_array], model_params)


def policy_logits(policy_params: Any, states: jnp.ndarray) -> jnp.ndarray:
    return apply_mlp(policy_params, states, activation=jax.nn.relu)


def q_values(q_params: Any, states: jnp.ndarray) -> jnp.ndarray:
    return apply_mlp(q_params, states, activation=jax.nn.relu)


def policy_distribution(policy_params: Any, states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = policy_logits(policy_params, states)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.exp(log_probs), log_probs


@jax.jit
def sample_actions(key: jax.Array, policy_params: Any, states: jnp.ndarray) -> jnp.ndarray:
    logits = policy_logits(policy_params, states)
    return jax.random.categorical(key, logits, axis=-1)


def dynamics_outputs(model_params: Any, states: jnp.ndarray, actions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    model_inputs = jnp.concatenate([states, actions[:, None].astype(jnp.float32)], axis=-1)

    def apply_single(single_model_params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        outputs = apply_mlp(single_model_params, model_inputs, activation=jax.nn.silu)
        mu, raw_log_var = jnp.split(outputs, 2, axis=-1)
        log_var = jnp.clip(raw_log_var, -10.0, 0.5)
        return mu, log_var

    return jax.vmap(apply_single)(model_params)


def make_world_model_train_step(world_optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(
        model_params: Any,
        opt_state: Any,
        batch_states: jnp.ndarray,
        batch_actions: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> tuple[Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def loss_fn(params: Any) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            mu, log_var = dynamics_outputs(params, batch_states, batch_actions)
            inv_var = jnp.exp(-log_var)
            squared_error = (mu - batch_targets[None, :, :]) ** 2
            per_model_nll = jnp.mean(squared_error * inv_var + log_var, axis=(1, 2))
            per_model_mse = jnp.mean(squared_error, axis=(1, 2))
            return per_model_nll.mean(), (per_model_nll, per_model_mse)

        (loss, (per_model_nll, per_model_mse)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_params)
        updates, opt_state = world_optimizer.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        return model_params, opt_state, loss, per_model_nll, per_model_mse

    return train_step


@jax.jit
def evaluate_world_model(
    model_params: Any,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    targets: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mu, log_var = dynamics_outputs(model_params, states, actions)
    inv_var = jnp.exp(-log_var)
    squared_error = (mu - targets[None, :, :]) ** 2
    per_model_nll = jnp.mean(squared_error * inv_var + log_var, axis=(1, 2))
    per_model_mse = jnp.mean(squared_error, axis=(1, 2))
    return per_model_nll, per_model_mse


def make_sac_update_fn(
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    gamma: float,
    tau: float,
    target_entropy: float,
    alpha_min: float,
):
    @jax.jit
    def sac_update(
        policy_params: Any,
        q1_params: Any,
        q2_params: Any,
        q1_target_params: Any,
        q2_target_params: Any,
        policy_opt_state: Any,
        q1_opt_state: Any,
        q2_opt_state: Any,
        log_alpha: jnp.ndarray,
        alpha_opt_state: Any,
        batch: tuple[jnp.ndarray, ...],
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, jnp.ndarray, Any, dict[str, jnp.ndarray]]:
        states, actions, rewards, next_states, dones = batch
        safe_alpha = jnp.maximum(jnp.exp(log_alpha), alpha_min)

        next_probs, next_log_probs = policy_distribution(policy_params, next_states)
        next_q = jnp.minimum(q_values(q1_target_params, next_states), q_values(q2_target_params, next_states))
        next_v = jnp.sum(next_probs * (next_q - safe_alpha * next_log_probs), axis=-1)
        q_targets = rewards + gamma * (1.0 - dones) * next_v

        def q_loss_fn(params: Any) -> jnp.ndarray:
            pred = jnp.take_along_axis(q_values(params, states), actions[:, None], axis=-1).squeeze(-1)
            return jnp.mean((pred - q_targets) ** 2)

        q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(q1_params)
        q1_updates, q1_opt_state = q_optimizer.update(q1_grads, q1_opt_state, q1_params)
        q1_params = optax.apply_updates(q1_params, q1_updates)

        q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(q2_params)
        q2_updates, q2_opt_state = q_optimizer.update(q2_grads, q2_opt_state, q2_params)
        q2_params = optax.apply_updates(q2_params, q2_updates)

        def policy_loss_fn(params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
            probs, log_probs = policy_distribution(params, states)
            q_min = jnp.minimum(q_values(q1_params, states), q_values(q2_params, states))
            entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
            loss = jnp.mean(jnp.sum(probs * (safe_alpha * log_probs - q_min), axis=-1))
            return loss, entropy

        (policy_loss, entropy), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
        policy_updates, policy_opt_state = policy_optimizer.update(policy_grads, policy_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        def alpha_loss_fn(current_log_alpha: jnp.ndarray) -> jnp.ndarray:
            return -(current_log_alpha * (target_entropy - entropy)).mean()

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        alpha_updates, alpha_opt_state = alpha_optimizer.update(alpha_grads, alpha_opt_state, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, alpha_updates)

        q1_target_params = optax.incremental_update(q1_params, q1_target_params, tau)
        q2_target_params = optax.incremental_update(q2_params, q2_target_params, tau)

        metrics = {
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.maximum(jnp.exp(log_alpha), alpha_min),
            "entropy": entropy,
        }
        return (
            policy_params,
            q1_params,
            q2_params,
            q1_target_params,
            q2_target_params,
            policy_opt_state,
            q1_opt_state,
            q2_opt_state,
            log_alpha,
            alpha_opt_state,
            metrics,
        )

    return sac_update


@partial(jax.jit, static_argnames=("rollout_length",))
def generate_model_rollouts(
    key: jax.Array,
    elite_model_params: Any,
    policy_params: Any,
    start_states: jnp.ndarray,
    rollout_length: int,
    death_penalty: float,
) -> dict[str, jnp.ndarray]:
    n_models = jax.tree_util.tree_leaves(elite_model_params)[0].shape[0]
    batch_size = start_states.shape[0]

    def scan_step(carry: tuple[jax.Array, jnp.ndarray, jnp.ndarray], _):
        rng, current_states, alive = carry
        rng, action_key, model_key, noise_key = jax.random.split(rng, 4)

        actions = sample_actions(action_key, policy_params, current_states)
        mu, log_var = dynamics_outputs(elite_model_params, current_states, actions)
        chosen_models = jax.random.randint(model_key, (batch_size,), 0, n_models)
        batch_indices = jnp.arange(batch_size)
        chosen_mu = mu[chosen_models, batch_indices]
        chosen_log_var = log_var[chosen_models, batch_indices]
        sampled_outputs = chosen_mu + jnp.exp(0.5 * chosen_log_var) * jax.random.normal(noise_key, chosen_mu.shape)

        delta_state = sampled_outputs[:, :NUM_FEATURES]
        rewards = sampled_outputs[:, -1]
        next_states = clamp_features(current_states + delta_state)
        dones = analytic_termination_from_features(next_states)
        rewards = jnp.where(dones, death_penalty, rewards)

        next_alive = alive & (~dones)
        transition = {
            "states": current_states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "mask": alive,
        }
        return (rng, next_states, next_alive), transition

    (_, _, _), rollout = jax.lax.scan(
        scan_step,
        (key, start_states, jnp.ones((batch_size,), dtype=bool)),
        xs=None,
        length=rollout_length,
    )
    return rollout


class AdaptiveRolloutScheduler:
    def __init__(self, min_length: int, max_length: int, warmup_steps: int, loss_ceiling: float, loss_floor: float):
        self.min_length = min_length
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.loss_ceiling = loss_ceiling
        self.loss_floor = loss_floor

    def get_length(self, total_steps: int, holdout_mse: float | None) -> int:
        if total_steps < self.warmup_steps or holdout_mse is None:
            return self.min_length
        if holdout_mse >= self.loss_ceiling:
            return self.min_length
        if holdout_mse <= self.loss_floor:
            return self.max_length

        progress = (self.loss_ceiling - holdout_mse) / (self.loss_ceiling - self.loss_floor)
        return int(round(self.min_length + progress * (self.max_length - self.min_length)))


def train_world_model(
    model_params: Any,
    opt_state: Any,
    real_buffer: ReplayBuffer,
    config: MBPOv6JAXConfig,
    train_step,
) -> tuple[Any, Any, dict[str, Any]]:
    if real_buffer.size < 512:
        raise ValueError("Need at least 512 real transitions before training the world model")

    sample_size = min(real_buffer.size, config.model_train_max_samples)
    if sample_size < real_buffer.size:
        selected = np.random.choice(real_buffer.size, size=sample_size, replace=False)
    else:
        selected = np.arange(real_buffer.size)

    states = jnp.asarray(real_buffer.states[selected])
    actions = jnp.asarray(real_buffer.actions[selected])
    next_states = jnp.asarray(real_buffer.next_states[selected])
    rewards = jnp.asarray(real_buffer.rewards[selected])
    targets = jnp.concatenate([next_states - states, rewards[:, None]], axis=-1)

    permutation = np.random.permutation(sample_size)
    holdout_size = max(256, int(sample_size * config.holdout_ratio))
    holdout_indices = permutation[:holdout_size]
    train_indices = permutation[holdout_size:]

    holdout_states = states[holdout_indices]
    holdout_actions = actions[holdout_indices]
    holdout_targets = targets[holdout_indices]

    best_params = model_params
    best_mean_nll = float("inf")
    best_metrics: tuple[np.ndarray, np.ndarray] | None = None
    stale_epochs = 0
    full_batches = max(1, len(train_indices) // config.model_batch_size)

    for _ in range(config.model_max_epochs):
        shuffled = np.random.permutation(train_indices)
        shuffled = shuffled[: full_batches * config.model_batch_size]

        for start in range(0, len(shuffled), config.model_batch_size):
            batch_indices = shuffled[start : start + config.model_batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_targets = targets[batch_indices]
            model_params, opt_state, _, _, _ = train_step(
                model_params,
                opt_state,
                batch_states,
                batch_actions,
                batch_targets,
            )

        holdout_nll, holdout_mse = evaluate_world_model(model_params, holdout_states, holdout_actions, holdout_targets)
        mean_holdout_nll = float(np.asarray(holdout_nll).mean())

        if mean_holdout_nll < best_mean_nll - 1e-6:
            best_mean_nll = mean_holdout_nll
            best_params = model_params
            best_metrics = (np.asarray(holdout_nll), np.asarray(holdout_mse))
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.model_patience:
                break

    model_params = best_params
    if best_metrics is None:
        holdout_nll, holdout_mse = evaluate_world_model(model_params, holdout_states, holdout_actions, holdout_targets)
        best_metrics = (np.asarray(holdout_nll), np.asarray(holdout_mse))

    per_model_holdout_nll, per_model_holdout_mse = best_metrics
    elite_indices = np.argsort(per_model_holdout_mse)[: config.elite_size].tolist()

    metrics = {
        "holdout_nll": float(np.mean(per_model_holdout_nll[elite_indices])),
        "holdout_mse": float(np.mean(per_model_holdout_mse[elite_indices])),
        "per_model_holdout_nll": [float(value) for value in per_model_holdout_nll],
        "per_model_holdout_mse": [float(value) for value in per_model_holdout_mse],
        "elite_indices": elite_indices,
    }
    return model_params, opt_state, metrics


def _make_training_batch(
    real_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    batch_size: int,
    real_fraction: float,
) -> tuple[jnp.ndarray, ...]:
    real_batch_size = max(1, int(round(batch_size * real_fraction)))
    model_batch_size = batch_size - real_batch_size

    if model_batch_size > 0 and model_buffer.size >= model_batch_size:
        real_batch = real_buffer.sample(real_batch_size)
        model_batch = model_buffer.sample(model_batch_size)
        return tuple(jnp.concatenate([real_item, model_item], axis=0) for real_item, model_item in zip(real_batch, model_batch))

    return real_buffer.sample(batch_size)


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)


def _checkpoint_payload(
    total_steps: int,
    total_episodes: int,
    config: MBPOv6JAXConfig,
    policy_params: Any,
    q1_params: Any,
    q2_params: Any,
    world_model_params: Any,
    elite_indices: list[int],
    raw_rewards: list[float],
    shaped_rewards: list[float],
    scores: list[int],
    model_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "config": asdict(config),
        "elite_indices": elite_indices,
        "raw_rewards": raw_rewards,
        "shaped_rewards": shaped_rewards,
        "scores": scores,
        "model_metrics": model_metrics,
        "policy_params": _tree_to_numpy(policy_params),
        "q1_params": _tree_to_numpy(q1_params),
        "q2_params": _tree_to_numpy(q2_params),
        "world_model_params": _tree_to_numpy(world_model_params),
    }


def save_checkpoint(
    config: MBPOv6JAXConfig,
    run_name: str,
    total_steps: int,
    total_episodes: int,
    policy_params: Any,
    q1_params: Any,
    q2_params: Any,
    world_model_params: Any,
    elite_indices: list[int],
    raw_rewards: list[float],
    shaped_rewards: list[float],
    scores: list[int],
    model_metrics: dict[str, Any] | None,
) -> Path:
    checkpoint_root = Path(config.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_root / f"{run_name}_ep{total_episodes:06d}_{timestamp}.pkl"
    with open(checkpoint_path, "wb") as handle:
        pickle.dump(
            _checkpoint_payload(
                total_steps,
                total_episodes,
                config,
                policy_params,
                q1_params,
                q2_params,
                world_model_params,
                elite_indices,
                raw_rewards,
                shaped_rewards,
                scores,
                model_metrics,
            ),
            handle,
        )

    existing = sorted(checkpoint_root.glob(f"{run_name}_ep*.pkl"))
    for old_path in existing[:-3]:
        old_path.unlink()
    return checkpoint_path


def train_mbpo_v6_jax(config: MBPOv6JAXConfig) -> MBPOv6JAXResult:
    set_seed(config.seed)
    run_name = config.run_name or f"mbpo_v6_jax_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(Path(config.tensorboard_root) / run_name))
    except Exception:
        writer = None

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_key = jax.random.split(rng)
    env_state, features = reset_env(reset_key, config.num_envs)

    real_buffer = ReplayBuffer(config.real_buffer_size)
    model_buffer = ReplayBuffer(config.model_buffer_size)

    rng, policy_key, q1_key, q2_key, world_key = jax.random.split(rng, 5)
    policy_params = init_mlp(policy_key, [NUM_FEATURES, config.policy_hidden_dim, config.policy_hidden_dim, 2])
    q1_params = init_mlp(q1_key, [NUM_FEATURES, config.q_hidden_dim, config.q_hidden_dim, 2])
    q2_params = init_mlp(q2_key, [NUM_FEATURES, config.q_hidden_dim, config.q_hidden_dim, 2])
    q1_target_params = q1_params
    q2_target_params = q2_params
    world_model_params = init_ensemble_mlp(
        world_key,
        config.ensemble_size,
        [NUM_FEATURES + 1, config.model_hidden_dim, config.model_hidden_dim, config.model_hidden_dim, 2 * (NUM_FEATURES + 1)],
    )

    policy_optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(config.sac_lr))
    q_optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(config.sac_lr))
    alpha_optimizer = optax.adam(config.sac_lr)
    world_optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(config.model_lr))

    policy_opt_state = policy_optimizer.init(policy_params)
    q1_opt_state = q_optimizer.init(q1_params)
    q2_opt_state = q_optimizer.init(q2_params)
    alpha_log = jnp.zeros((1,), dtype=jnp.float32)
    alpha_opt_state = alpha_optimizer.init(alpha_log)
    world_opt_state = world_optimizer.init(world_model_params)

    target_entropy = math.log(2.0) * config.target_entropy_ratio
    sac_update = make_sac_update_fn(
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        target_entropy=target_entropy,
        alpha_min=config.alpha_min,
    )
    world_model_train_step = make_world_model_train_step(world_optimizer)
    rollout_scheduler = AdaptiveRolloutScheduler(
        config.rollout_min_length,
        config.rollout_max_length,
        config.rollout_warmup_steps,
        config.rollout_loss_ceiling,
        config.rollout_loss_floor,
    )

    episode_raw_returns = np.zeros((config.num_envs,), dtype=np.float32)
    episode_shaped_returns = np.zeros((config.num_envs,), dtype=np.float32)
    episode_scores = np.zeros((config.num_envs,), dtype=np.int32)
    episode_lengths = np.zeros((config.num_envs,), dtype=np.int32)

    raw_reward_history: list[float] = []
    shaped_reward_history: list[float] = []
    score_history: list[int] = []
    episode_length_history: list[int] = []

    total_steps = 0
    total_episodes = 0
    best_raw_reward = float("-inf")
    best_score = 0
    elite_indices = list(range(config.elite_size))
    elite_model_params = select_ensemble_members(world_model_params, list(range(config.elite_size)))
    last_model_metrics: dict[str, Any] | None = None
    last_policy_metrics: dict[str, float] | None = None
    last_model_train_step = 0

    start_time = time_module.time()
    print(f"MBPO v6 JAX starting on backend={jax.default_backend()} with {config.cpu_threads} CPU threads")
    fallback_reason = os.environ.get("MBPO_V6_PLATFORM_FALLBACK_REASON")
    if fallback_reason:
        print(f"  Metal probe failed; using CPU fallback ({fallback_reason})")
    print(
        f"  envs={config.num_envs}, batch={config.batch_size}, ensemble={config.ensemble_size}/{config.elite_size}, "
        f"imagined_to_real={config.imagined_steps_per_real_step}:1"
    )

    while total_episodes < config.num_episodes:
        rng, action_key, step_key = jax.random.split(rng, 3)
        actions = np.asarray(sample_actions(action_key, policy_params, features), dtype=np.int32)
        env_state, next_features, raw_rewards, dones = env_step(step_key, env_state, jnp.asarray(actions), config.max_episode_steps)

        features_np = np.asarray(features)
        next_features_np = np.asarray(next_features)
        raw_rewards_np = np.asarray(raw_rewards)
        dones_np = np.asarray(dones)
        shaped_rewards = shape_rewards(
            states=features_np,
            next_states=next_features_np,
            raw_rewards=raw_rewards_np,
            dones=dones_np,
            gamma=config.gamma,
            alignment_scale=config.alignment_scale,
        )
        real_buffer.add_batch(features_np, actions, shaped_rewards, next_features_np, dones_np.astype(np.float32, copy=False))

        episode_raw_returns += raw_rewards_np
        episode_shaped_returns += shaped_rewards
        episode_scores += (raw_rewards_np > 0.5).astype(np.int32)
        episode_lengths += 1
        total_steps += config.num_envs

        done_indices = np.flatnonzero(dones_np)
        if done_indices.size:
            for env_index in done_indices:
                raw_return = float(episode_raw_returns[env_index])
                shaped_return = float(episode_shaped_returns[env_index])
                score = int(episode_scores[env_index])
                length = int(episode_lengths[env_index])

                raw_reward_history.append(raw_return)
                shaped_reward_history.append(shaped_return)
                score_history.append(score)
                episode_length_history.append(length)
                total_episodes += 1
                best_raw_reward = max(best_raw_reward, raw_return)
                best_score = max(best_score, score)

                if writer is not None:
                    writer.add_scalar("Episode/RawReward", raw_return, total_episodes)
                    writer.add_scalar("Episode/ShapedReward", shaped_return, total_episodes)
                    writer.add_scalar("Episode/Score", score, total_episodes)
                    writer.add_scalar("Episode/Length", length, total_episodes)

                episode_raw_returns[env_index] = 0.0
                episode_shaped_returns[env_index] = 0.0
                episode_scores[env_index] = 0
                episode_lengths[env_index] = 0

                if config.log_interval_episodes and total_episodes % config.log_interval_episodes == 0:
                    avg_raw = float(np.mean(raw_reward_history[-config.log_interval_episodes :]))
                    avg_shaped = float(np.mean(shaped_reward_history[-config.log_interval_episodes :]))
                    recent_scores = score_history[-100:]
                    avg_score_100 = float(np.mean(recent_scores)) if recent_scores else 0.0
                    max_score_100 = int(np.max(recent_scores)) if recent_scores else 0
                    holdout_mse = None if last_model_metrics is None else last_model_metrics["holdout_mse"]
                    rollout_length = rollout_scheduler.get_length(total_steps, holdout_mse)
                    elapsed = time_module.time() - start_time
                    episodes_per_second = total_episodes / max(elapsed, 1e-6)
                    alpha_value = float(last_policy_metrics["alpha"]) if last_policy_metrics is not None else config.alpha_min
                    print(
                        f"Ep {total_episodes:6d} | AvgRaw {avg_raw:6.2f} | AvgShaped {avg_shaped:7.2f} | "
                        f"AverageScore100 {avg_score_100:6.2f} | MaxScore100 {max_score_100:3d} | MaxScore {best_score:3d} | "
                        f"BestRaw {best_raw_reward:5.1f} | Alpha {alpha_value:5.3f} | RL {rollout_length:2d} | "
                        f"Real {real_buffer.size:7d} | Model {model_buffer.size:7d} | {episodes_per_second:5.2f} ep/s"
                    )

                    if writer is not None:
                        writer.add_scalar("Score/Average100", avg_score_100, total_episodes)
                        writer.add_scalar("Score/Max100", max_score_100, total_episodes)
                        writer.add_scalar("Score/MaxEver", best_score, total_episodes)

                if (
                    config.checkpoint_interval_episodes
                    and total_episodes % config.checkpoint_interval_episodes == 0
                ):
                    save_checkpoint(
                        config,
                        run_name,
                        total_steps,
                        total_episodes,
                        policy_params,
                        q1_params,
                        q2_params,
                        world_model_params,
                        elite_indices,
                        raw_reward_history,
                        shaped_reward_history,
                        score_history,
                        last_model_metrics,
                    )

                if total_episodes >= config.num_episodes:
                    break

            if total_episodes >= config.num_episodes:
                features = next_features
                break

            rng, reset_key = jax.random.split(rng)
            env_state, features = reset_done_envs(reset_key, env_state, jnp.asarray(dones_np))
        else:
            features = next_features

        min_model_train_samples = max(config.start_model_training_after, 512)
        should_train_model = (
            real_buffer.size >= min_model_train_samples
            and (total_steps - last_model_train_step) >= config.model_train_freq
        )
        if should_train_model:
            world_model_params, world_opt_state, last_model_metrics = train_world_model(
                world_model_params,
                world_opt_state,
                real_buffer,
                config,
                world_model_train_step,
            )
            elite_indices = list(last_model_metrics["elite_indices"])
            elite_model_params = select_ensemble_members(world_model_params, elite_indices)
            last_model_train_step = total_steps

            model_buffer.clear()
            rollout_length = rollout_scheduler.get_length(total_steps, last_model_metrics["holdout_mse"])
            target_imagined_steps = config.imagined_steps_per_real_step * config.model_train_freq
            rollout_starts = min(config.max_rollout_starts, math.ceil(target_imagined_steps / max(rollout_length, 1)))
            start_states = real_buffer.sample_states(rollout_starts)
            rng, rollout_key = jax.random.split(rng)
            rollout = generate_model_rollouts(
                rollout_key,
                elite_model_params,
                policy_params,
                start_states,
                rollout_length,
                config.death_penalty,
            )

            model_mask = np.asarray(rollout["mask"]).reshape(-1)
            model_buffer.add_batch(
                np.asarray(rollout["states"]).reshape(-1, NUM_FEATURES)[model_mask],
                np.asarray(rollout["actions"]).reshape(-1)[model_mask],
                np.asarray(rollout["rewards"]).reshape(-1)[model_mask],
                np.asarray(rollout["next_states"]).reshape(-1, NUM_FEATURES)[model_mask],
                np.asarray(rollout["dones"]).reshape(-1).astype(np.float32, copy=False)[model_mask],
            )

            if writer is not None:
                writer.add_scalar("WorldModel/HoldoutNLL", last_model_metrics["holdout_nll"], total_steps)
                writer.add_scalar("WorldModel/HoldoutMSE", last_model_metrics["holdout_mse"], total_steps)
                writer.add_scalar("WorldModel/RolloutLength", rollout_length, total_steps)
                writer.add_scalar("Buffers/ImaginedTransitions", int(model_mask.sum()), total_steps)

        if real_buffer.size >= config.start_policy_training_after:
            gradient_updates = max(1, int(round(config.gradient_updates_per_env_step * config.num_envs)))
            for _ in range(gradient_updates):
                batch = _make_training_batch(real_buffer, model_buffer, config.batch_size, config.real_fraction)
                (
                    policy_params,
                    q1_params,
                    q2_params,
                    q1_target_params,
                    q2_target_params,
                    policy_opt_state,
                    q1_opt_state,
                    q2_opt_state,
                    alpha_log,
                    alpha_opt_state,
                    metrics,
                ) = sac_update(
                    policy_params,
                    q1_params,
                    q2_params,
                    q1_target_params,
                    q2_target_params,
                    policy_opt_state,
                    q1_opt_state,
                    q2_opt_state,
                    alpha_log,
                    alpha_opt_state,
                    batch,
                )
                last_policy_metrics = {name: float(np.asarray(value)) for name, value in metrics.items()}

            if writer is not None and last_policy_metrics is not None:
                writer.add_scalar("SAC/Q1Loss", last_policy_metrics["q1_loss"], total_steps)
                writer.add_scalar("SAC/Q2Loss", last_policy_metrics["q2_loss"], total_steps)
                writer.add_scalar("SAC/PolicyLoss", last_policy_metrics["policy_loss"], total_steps)
                writer.add_scalar("SAC/Alpha", last_policy_metrics["alpha"], total_steps)
                writer.add_scalar("SAC/Entropy", last_policy_metrics["entropy"], total_steps)

        if writer is not None:
            writer.add_scalar("Buffers/RealSize", real_buffer.size, total_steps)
            writer.add_scalar("Buffers/ModelSize", model_buffer.size, total_steps)

    if writer is not None:
        writer.close()
    training_time = time_module.time() - start_time

    return MBPOv6JAXResult(
        config=config,
        run_name=run_name,
        backend=jax.default_backend(),
        cpu_threads=config.cpu_threads,
        policy_params=policy_params,
        q1_params=q1_params,
        q2_params=q2_params,
        world_model_params=world_model_params,
        elite_indices=elite_indices,
        raw_rewards=raw_reward_history,
        shaped_rewards=shaped_reward_history,
        scores=score_history,
        episode_lengths=episode_length_history,
        training_time_s=training_time,
        best_raw_reward=best_raw_reward,
        best_score=best_score,
        final_model_metrics=last_model_metrics,
    )


def plot_training_curves(
    result: MBPOv6JAXResult,
    window: int = 100,
    output_path: str | Path | None = None,
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(result.scores, linewidth=0.4, alpha=0.35, label="Score")
    if len(result.scores) >= window:
        moving_average = np.convolve(result.scores, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window - 1, window - 1 + len(moving_average)), moving_average, linewidth=2.0, label=f"{window}-episode avg")
    axes[0].set_title("MBPO v6 JAX Scores")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Pipes passed")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result.shaped_rewards, linewidth=0.4, alpha=0.35, label="Shaped reward")
    if len(result.shaped_rewards) >= window:
        moving_average = np.convolve(result.shaped_rewards, np.ones(window) / window, mode="valid")
        axes[1].plot(range(window - 1, window - 1 + len(moving_average)), moving_average, linewidth=2.0, label=f"{window}-episode avg")
    axes[1].set_title("MBPO v6 JAX Shaped Rewards")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Shaped reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    return figure


def save_training_artifacts(
    result: MBPOv6JAXResult,
    output_dir: str | Path = "saved_models/mbpo_v6_jax",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifact_payload = {
        "policy_params": _tree_to_numpy(result.policy_params),
        "q1_params": _tree_to_numpy(result.q1_params),
        "q2_params": _tree_to_numpy(result.q2_params),
        "world_model_params": _tree_to_numpy(result.world_model_params),
        "elite_indices": result.elite_indices,
    }
    with open(output_path / "model_bundle.pkl", "wb") as handle:
        pickle.dump(artifact_payload, handle)

    metadata = {
        "run_name": result.run_name,
        "backend": result.backend,
        "cpu_threads": result.cpu_threads,
        "episodes": len(result.raw_rewards),
        "avg_raw_reward_last_100": float(np.mean(result.raw_rewards[-100:])) if result.raw_rewards else 0.0,
        "avg_shaped_reward_last_100": float(np.mean(result.shaped_rewards[-100:])) if result.shaped_rewards else 0.0,
        "avg_score_last_100": float(np.mean(result.scores[-100:])) if result.scores else 0.0,
        "max_score_last_100": int(np.max(result.scores[-100:])) if result.scores else 0,
        "best_raw_reward": float(result.best_raw_reward) if result.raw_rewards else 0.0,
        "best_score": int(result.best_score),
        "training_time_s": float(result.training_time_s),
        "config": asdict(result.config),
        "final_model_metrics": result.final_model_metrics,
    }
    with open(output_path / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return output_path