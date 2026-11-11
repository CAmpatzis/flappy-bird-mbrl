"""Smoke test for the JAX-based MBPO v6 training stack."""

from __future__ import annotations

import numpy as np

from mbpo_v6_jax import MBPOv6JAXConfig, train_mbpo_v6_jax


def build_smoke_config() -> MBPOv6JAXConfig:
    return MBPOv6JAXConfig(
        num_episodes=24,
        num_envs=32,
        seed=7,
        max_episode_steps=64,
        batch_size=128,
        real_buffer_size=30_000,
        model_buffer_size=20_000,
        start_model_training_after=512,
        start_policy_training_after=512,
        gradient_updates_per_env_step=0.25,
        model_batch_size=128,
        model_max_epochs=3,
        model_patience=1,
        model_train_freq=128,
        model_train_max_samples=4_000,
        rollout_min_length=1,
        rollout_max_length=3,
        rollout_warmup_steps=512,
        imagined_steps_per_real_step=24,
        max_rollout_starts=256,
        checkpoint_interval_episodes=None,
        log_interval_episodes=8,
    )


def main() -> None:
    result = train_mbpo_v6_jax(build_smoke_config())

    assert len(result.raw_rewards) == 24
    assert len(result.shaped_rewards) == 24
    assert len(result.scores) == 24
    assert result.final_model_metrics is not None
    assert np.isfinite(result.final_model_metrics["holdout_mse"])
    assert all(np.isfinite(result.raw_rewards))
    assert all(np.isfinite(result.shaped_rewards))
    assert all(score >= 0 for score in result.scores)

    print("MBPO v6 JAX smoke test passed")
    print(f"  backend: {result.backend}")
    print(f"  cpu_threads: {result.cpu_threads}")
    print(f"  avg raw last 8: {np.mean(result.raw_rewards[-8:]):.3f}")
    print(f"  avg score last 8: {np.mean(result.scores[-8:]):.3f}")
    print(f"  max score: {result.best_score}")
    print(f"  holdout MSE: {result.final_model_metrics['holdout_mse']:.6f}")


if __name__ == "__main__":
    main()