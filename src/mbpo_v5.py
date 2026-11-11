"""MBPO v5 for Flappy Bird.

This version intentionally keeps the full pipeline in a compact, file-backed
implementation rather than another notebook-only fork:

- Headless feature-only environment
- Discrete SAC policy backbone
- 7-model probabilistic ensemble dynamics model with 5 elites
- Validation-driven rollout horizon schedule
- Strict fixed-shape replay buffers for both real and imagined data
"""

from __future__ import annotations

import copy
import json
import math
import random
import time as time_module
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mbpo_v5_env import (
    BASE_Y,
    NUM_FEATURES,
    PIPE_GAP,
    PIPE_W,
    PLAYER_H,
    PLAYER_W,
    SCREEN_H,
    SCREEN_W,
    HeadlessFlappyBirdVecEnv,
    feature_potential,
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _default_device()
torch.set_float32_matmul_precision("high")


@dataclass(slots=True)
class MBPOv5Config:
    num_episodes: int = 20_000
    num_envs: int = 64
    seed: int = 7
    max_episode_steps: int | None = None

    real_buffer_size: int = 1_000_000
    model_buffer_size: int = 500_000
    batch_size: int = 256

    gamma: float = 0.99
    sac_lr: float = 3e-4
    model_lr: float = 1e-3
    tau: float = 0.005
    alpha_min: float = 0.01
    target_entropy_ratio: float = 0.9
    policy_hidden_dim: int = 256
    model_hidden_dim: int = 256

    start_model_training_after: int = 2_048
    start_policy_training_after: int = 4_096
    real_fraction: float = 0.05
    gradient_updates_per_env_step: float = 0.5

    ensemble_size: int = 7
    elite_size: int = 5
    holdout_ratio: float = 0.2
    model_batch_size: int = 512
    model_max_epochs: int = 30
    model_patience: int = 5
    model_train_freq: int = 512
    model_train_max_samples: int = 50_000

    rollout_min_length: int = 1
    rollout_max_length: int = 10
    rollout_warmup_steps: int = 5_000
    rollout_loss_ceiling: float = 2e-2
    rollout_loss_floor: float = 2e-3
    imagined_steps_per_real_step: int = 400
    max_rollout_starts: int = 20_000

    alignment_scale: float = 2.0
    death_penalty: float = -1.0

    tensorboard_root: str = "runs"
    checkpoint_dir: str = "checkpoints/mbpo_v5"
    checkpoint_interval_episodes: int | None = 5_000
    log_interval_episodes: int = 100
    run_name: str | None = None


@dataclass
class MBPOv5Result:
    config: MBPOv5Config
    run_name: str
    device: str
    sac: "DiscreteSAC"
    world_models: list["ProbabilisticDynamicsModel"]
    raw_rewards: list[float] = field(default_factory=list)
    shaped_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    training_time_s: float = 0.0
    best_raw_reward: float = float("-inf")
    final_model_metrics: dict[str, Any] | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp_features(features: torch.Tensor) -> torch.Tensor:
    clamped = features.clone()
    clamped[..., 0] = clamped[..., 0].clamp(-0.20, 1.20)
    clamped[..., 1] = clamped[..., 1].clamp(-1.20, 1.20)
    clamped[..., 2::2] = clamped[..., 2::2].clamp(-0.50, 2.00)
    clamped[..., 3::2] = clamped[..., 3::2].clamp(0.0, 1.0)
    return clamped


def analytic_termination_from_features(features: torch.Tensor) -> torch.Tensor:
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


def shape_rewards(
    states: np.ndarray,
    next_states: np.ndarray,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    alignment_scale: float,
) -> np.ndarray:
    phi_state = feature_potential(states)
    phi_next = feature_potential(next_states)
    phi_next = np.where(dones, 0.0, phi_next)
    shaping = alignment_scale * (gamma * phi_next - phi_state)
    return (raw_rewards + shaping).astype(np.float32, copy=False)


class ReplayBuffer:
    """Strict fixed-shape replay buffer shared by real and model data."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

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
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must have shape [B, {self.state_dim}], got {states.shape}")
        if next_states.shape != states.shape:
            raise ValueError(f"next_states shape {next_states.shape} does not match states {states.shape}")

        batch_size = states.shape[0]
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

    def sample(self, batch_size: int, device: torch.device = DEVICE) -> tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_states[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
        )

    def sample_states(self, count: int, device: torch.device = DEVICE) -> torch.Tensor:
        if self.size == 0:
            raise ValueError("Cannot sample states from an empty replay buffer")
        indices = np.random.randint(0, self.size, size=count)
        return torch.from_numpy(self.states[indices]).to(device)


class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim: int = NUM_FEATURES, hidden_dim: int = 256, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample_actions(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        probs, _ = self(states)
        if deterministic:
            return probs.argmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = NUM_FEATURES, hidden_dim: int = 256, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class DiscreteSAC:
    def __init__(
        self,
        state_dim: int = NUM_FEATURES,
        hidden_dim: int = 256,
        n_actions: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy_ratio: float = 0.9,
        alpha_min: float = 0.01,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.alpha_min = float(alpha_min)

        self.policy = CategoricalPolicy(state_dim, hidden_dim, n_actions).to(device)
        self.q1 = QNetwork(state_dim, hidden_dim, n_actions).to(device)
        self.q2 = QNetwork(state_dim, hidden_dim, n_actions).to(device)
        self.q1_target = QNetwork(state_dim, hidden_dim, n_actions).to(device)
        self.q2_target = QNetwork(state_dim, hidden_dim, n_actions).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.target_entropy = math.log(n_actions) * target_entropy_ratio
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> float:
        return max(self.log_alpha.exp().item(), self.alpha_min)

    def select_actions(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            actions = self.policy.sample_actions(state_tensor, deterministic=deterministic)
        return actions.cpu().numpy().astype(np.int64, copy=False)

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        alpha_floor = torch.tensor(self.alpha_min, device=self.device)
        alpha = torch.maximum(self.log_alpha.exp(), alpha_floor)

        with torch.no_grad():
            next_probs, next_log_probs = self.policy(next_states)
            next_q = torch.min(self.q1_target(next_states), self.q2_target(next_states))
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
            q_target = rewards + self.gamma * (1.0 - dones) * next_v

        q1_pred = self.q1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = self.q2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
        self.q2_optimizer.step()

        probs, log_probs = self.policy(states)
        with torch.no_grad():
            q_values = torch.min(self.q1(states), self.q2(states))
        policy_loss = (probs * (alpha.detach() * log_probs - q_values)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.policy_optimizer.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy)).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.lerp_(param.data, self.tau)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.lerp_(param.data, self.tau)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha),
            "entropy": float(entropy.item()),
        }


class ProbabilisticDynamicsModel(nn.Module):
    def __init__(self, state_dim: int = NUM_FEATURES, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + 1
        output_dim = state_dim + 1
        self.state_dim = state_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_var_head = nn.Linear(hidden_dim, output_dim)
        self.max_log_var = nn.Parameter(torch.full((output_dim,), 0.5))
        self.min_log_var = nn.Parameter(torch.full((output_dim,), -10.0))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_input = actions.unsqueeze(-1).float() if actions.ndim == 1 else actions.float()
        hidden = self.backbone(torch.cat([states, action_input], dim=-1))
        mu = self.mu_head(hidden)
        log_var = self.log_var_head(hidden)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mu, log_var

    def sample_prediction(
        self, states: torch.Tensor, actions: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self(states, actions)
        if deterministic:
            prediction = mu
        else:
            std = torch.exp(0.5 * log_var)
            prediction = mu + torch.randn_like(std) * std

        delta_state = prediction[..., : self.state_dim]
        reward = prediction[..., -1]
        next_states = clamp_features(states + delta_state)
        return next_states, reward


class EnsembleDynamicsTrainer:
    def __init__(
        self,
        models: list[ProbabilisticDynamicsModel],
        elite_size: int,
        lr: float,
        holdout_ratio: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        max_samples: int,
        device: torch.device = DEVICE,
    ):
        if elite_size <= 0 or elite_size > len(models):
            raise ValueError("elite_size must be in [1, len(models)]")

        self.models = models
        self.elite_size = elite_size
        self.holdout_ratio = holdout_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_samples = max_samples
        self.device = device
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) for model in models]
        self.elite_indices = list(range(elite_size))

    @staticmethod
    def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-log_var)
        squared_error = (mu - target) ** 2
        return (squared_error * inv_var + log_var).mean()

    def train(self, real_buffer: ReplayBuffer) -> dict[str, Any]:
        if real_buffer.size < 512:
            raise ValueError("Need at least 512 real transitions before training the world model")

        total_size = real_buffer.size
        sample_size = min(total_size, self.max_samples)
        if sample_size < total_size:
            subset = np.random.choice(total_size, size=sample_size, replace=False)
        else:
            subset = np.arange(total_size)

        states = torch.from_numpy(real_buffer.states[subset]).to(self.device)
        actions = torch.from_numpy(real_buffer.actions[subset]).to(self.device)
        next_states = torch.from_numpy(real_buffer.next_states[subset]).to(self.device)
        rewards = torch.from_numpy(real_buffer.rewards[subset]).to(self.device)
        targets = torch.cat([next_states - states, rewards.unsqueeze(-1)], dim=-1)

        permutation = np.random.permutation(sample_size)
        holdout_size = max(256, int(sample_size * self.holdout_ratio))
        holdout_indices = permutation[:holdout_size]
        train_indices = permutation[holdout_size:]

        holdout_states = states[holdout_indices]
        holdout_actions = actions[holdout_indices]
        holdout_targets = targets[holdout_indices]

        best_nlls: list[float] = []
        best_mses: list[float] = []

        for model, optimizer in zip(self.models, self.optimizers):
            bootstrap_indices = np.random.choice(train_indices, size=len(train_indices), replace=True)
            best_state = copy.deepcopy(model.state_dict())
            best_nll = float("inf")
            best_mse = float("inf")
            stale_epochs = 0

            for _ in range(self.max_epochs):
                model.train()
                order = np.random.permutation(len(bootstrap_indices))
                for start in range(0, len(order), self.batch_size):
                    batch_order = order[start : start + self.batch_size]
                    batch_indices = bootstrap_indices[batch_order]
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_targets = targets[batch_indices]

                    mu, log_var = model(batch_states, batch_actions)
                    loss = self.gaussian_nll(mu, log_var, batch_targets)
                    loss = loss + 1e-4 * (model.max_log_var.sum() - model.min_log_var.sum())

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    mu, log_var = model(holdout_states, holdout_actions)
                    holdout_nll = float(self.gaussian_nll(mu, log_var, holdout_targets).item())
                    holdout_mse = float(F.mse_loss(mu, holdout_targets).item())

                if holdout_nll < best_nll:
                    best_nll = holdout_nll
                    best_mse = holdout_mse
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.patience:
                        break

            model.load_state_dict(best_state)
            best_nlls.append(best_nll)
            best_mses.append(best_mse)

        ranked = sorted(range(len(self.models)), key=lambda index: best_nlls[index])
        self.elite_indices = ranked[: self.elite_size]
        elite_nll = [best_nlls[index] for index in self.elite_indices]
        elite_mse = [best_mses[index] for index in self.elite_indices]

        return {
            "holdout_nll": float(np.mean(elite_nll)),
            "holdout_mse": float(np.mean(elite_mse)),
            "per_model_holdout_nll": [float(value) for value in best_nlls],
            "per_model_holdout_mse": [float(value) for value in best_mses],
            "elite_indices": list(self.elite_indices),
        }


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
        rollout_length = self.min_length + progress * (self.max_length - self.min_length)
        return int(round(rollout_length))


class ModelRolloutGenerator:
    def __init__(
        self,
        world_models: list[ProbabilisticDynamicsModel],
        policy: CategoricalPolicy,
        model_buffer: ReplayBuffer,
        elite_indices: list[int],
        death_penalty: float,
        device: torch.device = DEVICE,
    ):
        self.world_models = world_models
        self.policy = policy
        self.model_buffer = model_buffer
        self.elite_indices = elite_indices
        self.death_penalty = death_penalty
        self.device = device

    def generate(self, start_states: torch.Tensor, rollout_length: int) -> int:
        states = start_states.clone()
        alive = torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
        total_added = 0

        for _ in range(rollout_length):
            if not bool(alive.any()):
                break

            current_states = states[alive]
            with torch.no_grad():
                actions = self.policy.sample_actions(current_states)
                next_states = torch.empty_like(current_states)
                rewards = torch.empty(current_states.shape[0], device=self.device)

                model_choices = torch.randint(0, len(self.elite_indices), (current_states.shape[0],), device=self.device)
                for local_model_idx, elite_model_idx in enumerate(self.elite_indices):
                    selection = model_choices == local_model_idx
                    if not bool(selection.any()):
                        continue
                    predicted_next, predicted_reward = self.world_models[elite_model_idx].sample_prediction(
                        current_states[selection], actions[selection]
                    )
                    next_states[selection] = predicted_next
                    rewards[selection] = predicted_reward

                dones = analytic_termination_from_features(next_states)
                rewards = rewards.clone()
                rewards[dones] = self.death_penalty

            self.model_buffer.add_batch(
                current_states.detach().cpu().numpy(),
                actions.detach().cpu().numpy(),
                rewards.detach().cpu().numpy(),
                next_states.detach().cpu().numpy(),
                dones.detach().cpu().numpy().astype(np.float32, copy=False),
            )
            total_added += current_states.shape[0]

            alive_indices = alive.nonzero(as_tuple=True)[0]
            states[alive_indices] = next_states
            updated_alive = alive.clone()
            updated_alive[alive_indices[dones]] = False
            alive = updated_alive

        return total_added


def _checkpoint_payload(
    total_steps: int,
    total_episodes: int,
    sac: DiscreteSAC,
    world_models: list[ProbabilisticDynamicsModel],
    raw_rewards: list[float],
    shaped_rewards: list[float],
    model_metrics: dict[str, Any] | None,
    config: MBPOv5Config,
) -> dict[str, Any]:
    return {
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "config": asdict(config),
        "raw_rewards": raw_rewards,
        "shaped_rewards": shaped_rewards,
        "model_metrics": model_metrics,
        "sac": {
            "policy": sac.policy.state_dict(),
            "q1": sac.q1.state_dict(),
            "q2": sac.q2.state_dict(),
            "q1_target": sac.q1_target.state_dict(),
            "q2_target": sac.q2_target.state_dict(),
            "log_alpha": sac.log_alpha.detach().cpu(),
        },
        "world_models": [model.state_dict() for model in world_models],
    }


def save_checkpoint(
    config: MBPOv5Config,
    run_name: str,
    total_steps: int,
    total_episodes: int,
    sac: DiscreteSAC,
    world_models: list[ProbabilisticDynamicsModel],
    raw_rewards: list[float],
    shaped_rewards: list[float],
    model_metrics: dict[str, Any] | None,
) -> Path:
    checkpoint_root = Path(config.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_root / f"{run_name}_ep{total_episodes:06d}_{timestamp}.pt"
    torch.save(
        _checkpoint_payload(
            total_steps,
            total_episodes,
            sac,
            world_models,
            raw_rewards,
            shaped_rewards,
            model_metrics,
            config,
        ),
        checkpoint_path,
    )

    existing = sorted(checkpoint_root.glob(f"{run_name}_ep*.pt"))
    for old_path in existing[:-3]:
        old_path.unlink()

    return checkpoint_path


def _make_training_batch(
    real_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    batch_size: int,
    real_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    real_batch_size = max(1, int(round(batch_size * real_fraction)))
    model_batch_size = batch_size - real_batch_size

    if model_batch_size > 0 and model_buffer.size >= model_batch_size:
        real_batch = real_buffer.sample(real_batch_size, device=device)
        model_batch = model_buffer.sample(model_batch_size, device=device)
        return tuple(torch.cat([real_item, model_item], dim=0) for real_item, model_item in zip(real_batch, model_batch))

    return real_buffer.sample(batch_size, device=device)


def train_mbpo_v5(config: MBPOv5Config) -> MBPOv5Result:
    set_seed(config.seed)

    run_name = config.run_name or f"mbpo_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(Path(config.tensorboard_root) / run_name))

    env = HeadlessFlappyBirdVecEnv(config.num_envs, seed=config.seed, max_episode_steps=config.max_episode_steps)
    real_buffer = ReplayBuffer(config.real_buffer_size, NUM_FEATURES)
    model_buffer = ReplayBuffer(config.model_buffer_size, NUM_FEATURES)

    sac = DiscreteSAC(
        state_dim=NUM_FEATURES,
        hidden_dim=config.policy_hidden_dim,
        lr=config.sac_lr,
        gamma=config.gamma,
        tau=config.tau,
        target_entropy_ratio=config.target_entropy_ratio,
        alpha_min=config.alpha_min,
        device=DEVICE,
    )

    world_models = [
        ProbabilisticDynamicsModel(state_dim=NUM_FEATURES, hidden_dim=config.model_hidden_dim).to(DEVICE)
        for _ in range(config.ensemble_size)
    ]
    ensemble_trainer = EnsembleDynamicsTrainer(
        models=world_models,
        elite_size=config.elite_size,
        lr=config.model_lr,
        holdout_ratio=config.holdout_ratio,
        batch_size=config.model_batch_size,
        max_epochs=config.model_max_epochs,
        patience=config.model_patience,
        max_samples=config.model_train_max_samples,
        device=DEVICE,
    )
    rollout_scheduler = AdaptiveRolloutScheduler(
        min_length=config.rollout_min_length,
        max_length=config.rollout_max_length,
        warmup_steps=config.rollout_warmup_steps,
        loss_ceiling=config.rollout_loss_ceiling,
        loss_floor=config.rollout_loss_floor,
    )
    rollout_generator = ModelRolloutGenerator(
        world_models=world_models,
        policy=sac.policy,
        model_buffer=model_buffer,
        elite_indices=ensemble_trainer.elite_indices,
        death_penalty=config.death_penalty,
        device=DEVICE,
    )

    states = env.features().copy()
    episode_raw_returns = np.zeros(config.num_envs, dtype=np.float32)
    episode_shaped_returns = np.zeros(config.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(config.num_envs, dtype=np.int32)

    raw_reward_history: list[float] = []
    shaped_reward_history: list[float] = []
    episode_length_history: list[int] = []

    total_steps = 0
    total_episodes = 0
    best_raw_reward = float("-inf")
    last_policy_metrics: dict[str, float] | None = None
    last_model_metrics: dict[str, Any] | None = None
    last_model_train_step = 0

    start_time = time_module.time()
    print(f"MBPO v5 starting on {DEVICE} with {config.num_envs} headless envs")
    print(
        f"  ensemble={config.ensemble_size}/{config.elite_size} elites, "
        f"imagined_to_real={config.imagined_steps_per_real_step}:1, batch={config.batch_size}"
    )

    while total_episodes < config.num_episodes:
        actions = sac.select_actions(states)
        next_states, raw_rewards, dones = env.step(actions)
        shaped_rewards = shape_rewards(
            states=states,
            next_states=next_states,
            raw_rewards=raw_rewards,
            dones=dones,
            gamma=config.gamma,
            alignment_scale=config.alignment_scale,
        )

        real_buffer.add_batch(states, actions, shaped_rewards, next_states, dones.astype(np.float32))

        episode_raw_returns += raw_rewards
        episode_shaped_returns += shaped_rewards
        episode_lengths += 1

        total_steps += config.num_envs

        done_indices = np.flatnonzero(dones)
        if done_indices.size:
            for env_index in done_indices:
                raw_return = float(episode_raw_returns[env_index])
                shaped_return = float(episode_shaped_returns[env_index])
                length = int(episode_lengths[env_index])

                raw_reward_history.append(raw_return)
                shaped_reward_history.append(shaped_return)
                episode_length_history.append(length)
                total_episodes += 1
                best_raw_reward = max(best_raw_reward, raw_return)

                writer.add_scalar("Episode/RawReward", raw_return, total_episodes)
                writer.add_scalar("Episode/ShapedReward", shaped_return, total_episodes)
                writer.add_scalar("Episode/Length", length, total_episodes)

                episode_raw_returns[env_index] = 0.0
                episode_shaped_returns[env_index] = 0.0
                episode_lengths[env_index] = 0

                if config.log_interval_episodes and total_episodes % config.log_interval_episodes == 0:
                    avg_raw = float(np.mean(raw_reward_history[-config.log_interval_episodes :]))
                    avg_shaped = float(np.mean(shaped_reward_history[-config.log_interval_episodes :]))
                    holdout_mse = None if last_model_metrics is None else last_model_metrics["holdout_mse"]
                    rollout_length = rollout_scheduler.get_length(total_steps, holdout_mse)
                    elapsed = time_module.time() - start_time
                    episodes_per_second = total_episodes / max(elapsed, 1e-6)
                    print(
                        f"Ep {total_episodes:6d} | AvgRaw {avg_raw:6.2f} | AvgShaped {avg_shaped:7.2f} | "
                        f"Best {best_raw_reward:5.1f} | Alpha {sac.alpha:5.3f} | RL {rollout_length:2d} | "
                        f"Real {real_buffer.size:7d} | Model {model_buffer.size:7d} | {episodes_per_second:5.2f} ep/s"
                    )

                if (
                    config.checkpoint_interval_episodes
                    and total_episodes % config.checkpoint_interval_episodes == 0
                ):
                    save_checkpoint(
                        config,
                        run_name,
                        total_steps,
                        total_episodes,
                        sac,
                        world_models,
                        raw_reward_history,
                        shaped_reward_history,
                        last_model_metrics,
                    )

                if total_episodes >= config.num_episodes:
                    break

            reset_mask = np.zeros(config.num_envs, dtype=np.bool_)
            reset_mask[done_indices] = True
            reset_states = env.reset(reset_mask)
            next_states_for_loop = next_states.copy()
            next_states_for_loop[done_indices] = reset_states
            states = next_states_for_loop
        else:
            states = next_states

        min_model_train_samples = max(config.start_model_training_after, 512)
        should_train_model = (
            real_buffer.size >= min_model_train_samples
            and (total_steps - last_model_train_step) >= config.model_train_freq
        )
        if should_train_model:
            last_model_metrics = ensemble_trainer.train(real_buffer)
            rollout_generator.elite_indices = ensemble_trainer.elite_indices
            last_model_train_step = total_steps

            writer.add_scalar("WorldModel/HoldoutNLL", last_model_metrics["holdout_nll"], total_steps)
            writer.add_scalar("WorldModel/HoldoutMSE", last_model_metrics["holdout_mse"], total_steps)

            model_buffer.clear()
            rollout_length = rollout_scheduler.get_length(total_steps, last_model_metrics["holdout_mse"])
            writer.add_scalar("WorldModel/RolloutLength", rollout_length, total_steps)

            target_imagined_steps = config.imagined_steps_per_real_step * max(config.model_train_freq, total_steps - last_model_train_step + config.model_train_freq)
            rollout_starts = math.ceil(target_imagined_steps / max(rollout_length, 1))
            rollout_starts = min(config.max_rollout_starts, rollout_starts)
            start_states = real_buffer.sample_states(rollout_starts, device=DEVICE)
            imagined_transitions = rollout_generator.generate(start_states, rollout_length)
            writer.add_scalar("Buffers/ImaginedTransitions", imagined_transitions, total_steps)

        if real_buffer.size >= config.start_policy_training_after:
            gradient_updates = max(1, int(round(config.gradient_updates_per_env_step * config.num_envs)))
            for _ in range(gradient_updates):
                batch = _make_training_batch(
                    real_buffer=real_buffer,
                    model_buffer=model_buffer,
                    batch_size=config.batch_size,
                    real_fraction=config.real_fraction,
                    device=DEVICE,
                )
                last_policy_metrics = sac.update(batch)

            if last_policy_metrics is not None:
                writer.add_scalar("SAC/Q1Loss", last_policy_metrics["q1_loss"], total_steps)
                writer.add_scalar("SAC/Q2Loss", last_policy_metrics["q2_loss"], total_steps)
                writer.add_scalar("SAC/PolicyLoss", last_policy_metrics["policy_loss"], total_steps)
                writer.add_scalar("SAC/Alpha", last_policy_metrics["alpha"], total_steps)
                writer.add_scalar("SAC/Entropy", last_policy_metrics["entropy"], total_steps)

        writer.add_scalar("Buffers/RealSize", real_buffer.size, total_steps)
        writer.add_scalar("Buffers/ModelSize", model_buffer.size, total_steps)

    writer.close()
    training_time = time_module.time() - start_time

    return MBPOv5Result(
        config=config,
        run_name=run_name,
        device=str(DEVICE),
        sac=sac,
        world_models=world_models,
        raw_rewards=raw_reward_history,
        shaped_rewards=shaped_reward_history,
        episode_lengths=episode_length_history,
        training_time_s=training_time,
        best_raw_reward=best_raw_reward,
        final_model_metrics=last_model_metrics,
    )


def _result_summary(result: MBPOv5Result) -> dict[str, Any]:
    avg_raw_last_100 = float(np.mean(result.raw_rewards[-100:])) if result.raw_rewards else 0.0
    avg_shaped_last_100 = float(np.mean(result.shaped_rewards[-100:])) if result.shaped_rewards else 0.0
    return {
        "run_name": result.run_name,
        "device": result.device,
        "episodes": len(result.raw_rewards),
        "avg_raw_reward_last_100": avg_raw_last_100,
        "avg_shaped_reward_last_100": avg_shaped_last_100,
        "best_raw_reward": float(result.best_raw_reward) if result.raw_rewards else 0.0,
        "training_time_s": float(result.training_time_s),
        "config": asdict(result.config),
        "final_model_metrics": result.final_model_metrics,
    }


def save_training_artifacts(result: MBPOv5Result, output_dir: str | Path = "saved_models/mbpo_v5") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(result.sac.policy.state_dict(), output_path / "policy_model.pt")
    torch.save(result.sac.q1.state_dict(), output_path / "q1_model.pt")
    torch.save(result.sac.q2.state_dict(), output_path / "q2_model.pt")
    for model_index, model in enumerate(result.world_models):
        torch.save(model.state_dict(), output_path / f"world_model_{model_index}.pt")

    with open(output_path / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(_result_summary(result), handle, indent=2)

    return output_path


def plot_training_curves(
    result: MBPOv5Result,
    window: int = 100,
    output_path: str | Path | None = None,
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(result.raw_rewards, linewidth=0.4, alpha=0.35, label="Raw reward")
    if len(result.raw_rewards) >= window:
        moving_average = np.convolve(result.raw_rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window - 1, window - 1 + len(moving_average)), moving_average, linewidth=2.0, label=f"{window}-episode avg")
    axes[0].set_title("MBPO v5 Raw Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Raw reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result.shaped_rewards, linewidth=0.4, alpha=0.35, label="Shaped reward")
    if len(result.shaped_rewards) >= window:
        moving_average = np.convolve(result.shaped_rewards, np.ones(window) / window, mode="valid")
        axes[1].plot(range(window - 1, window - 1 + len(moving_average)), moving_average, linewidth=2.0, label=f"{window}-episode avg")
    axes[1].set_title("MBPO v5 Shaped Rewards")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Shaped reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    return figure