# Flappy Bird RL Portfolio (PPO vs MBPO)

This repository is a cleaned, employer-facing version of my Flappy Bird reinforcement learning project.

It contains:
- End-to-end PPO training implementation in PyTorch
- Model-based RL iterations (MBPO v5 and MBPO v6 JAX)
- Original MBPO notebook that achieved the best recorded performance in this workspace
- Reproducibility scripts, smoke tests, and benchmark evidence

## Best Recorded Result

Based on preserved experiment metadata:

| Model | Episodes | Average Reward (last 100) | Max Reward |
|---|---:|---:|---:|
| MBPO (best run) | 50,000 | 12.265 | 61.8 |
| PPO (reference) | 50,000 | 10.94 (mean eval) | 20.0 |

Source files:
- results/best_mbpo_metadata.json
- results/ppo_reference_metadata.json
- results/benchmark_summary.json

## Benchmark Figure

![MBPO vs PPO benchmark panel](results/fig4_combined_panel.png)

Why the reward values appear in vertical "columns":
- Flappy Bird rewards are partly discrete events (especially pipe passes and terminal penalty), not fully continuous outcomes.
- With frame skipping and fixed per-step shaping, many episodes collapse to similar totals.
- That naturally creates stacked bands/columns in the distribution plot, which is a useful signal of environment reward granularity.

## Repository Structure

- src/: training/evaluation code and smoke tests
- notebooks/: original experimental notebooks
- flappy_bird_repo/: game environment dependency used by agents
- results/: benchmark artifacts and metadata
- scripts/: utility scripts for result reporting

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run PPO training:

```bash
python src/flappy_bird_ppo.py
```

4. Run smoke tests:

```bash
python src/test_mbpo_v5_smoke.py
python src/test_mbpo_v6_smoke.py
```

5. Generate a benchmark report from metadata:

```bash
python scripts/summarize_results.py
```

## Notes for Recruiters

This project demonstrates:
- Practical deep RL engineering (PPO, world models, vectorized environments)
- Iterative experimentation across multiple MBPO versions
- Reproducibility discipline via checkpoints, metrics, and smoke tests
- Cross-framework work in PyTorch and JAX

## Attribution

The game implementation under flappy_bird_repo/ is adapted from an external Flappy Bird RL codebase included in this workspace, with its own LICENSE retained.
