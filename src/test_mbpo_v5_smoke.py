"""Smoke test for the MBPO v5 training stack.

This is intentionally short and only verifies that the end-to-end pipeline runs:
real data collection, world-model fitting, model rollouts, and SAC updates.
It is not a performance benchmark.
"""

from __future__ import annotations

import numpy as np

from mbpo_v5 import MBPOv5Config, train_mbpo_v5


def build_smoke_config() -> MBPOv5Config:
    return MBPOv5Config(
        num_episodes=32,
        num_envs=16,
        seed=7,
        max_episode_steps=64,
        real_buffer_size=20_000,
        model_buffer_size=20_000,
        batch_size=64,
        start_model_training_after=256,
        start_policy_training_after=256,
        gradient_updates_per_env_step=0.25,
        ensemble_size=7,
        elite_size=5,
        model_batch_size=128,
        model_max_epochs=2,
        model_patience=1,
        model_train_freq=128,
        model_train_max_samples=4_000,
        rollout_min_length=1,
        rollout_max_length=3,
        rollout_warmup_steps=256,
        imagined_steps_per_real_step=20,
        max_rollout_starts=256,
        checkpoint_interval_episodes=None,
        log_interval_episodes=8,
    )


def main() -> None:
    config = build_smoke_config()
    result = train_mbpo_v5(config)

    assert len(result.raw_rewards) == config.num_episodes
    assert len(result.shaped_rewards) == config.num_episodes
    assert all(np.isfinite(result.raw_rewards))
    assert all(np.isfinite(result.shaped_rewards))
    assert result.final_model_metrics is not None
    assert np.isfinite(result.final_model_metrics["holdout_mse"])

    print("MBPO v5 smoke test passed")
    print(f"  episodes: {len(result.raw_rewards)}")
    print(f"  avg raw last 8: {np.mean(result.raw_rewards[-8:]):.3f}")
    print(f"  best raw reward: {np.max(result.raw_rewards):.3f}")
    print(f"  holdout MSE: {result.final_model_metrics['holdout_mse']:.6f}")


if __name__ == "__main__":
    main()