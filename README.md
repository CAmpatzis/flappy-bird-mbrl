# flappy-bird-mbrl

Model-based and model-free reinforcement learning benchmark on Flappy Bird, curated as a publication-quality engineering portfolio.

## Abstract

This project benchmarks model-based reinforcement learning (MBRL/MBPO variants) against PPO on Flappy Bird under a shared environment and comparable training budgets. The repository is curated to highlight reproducible experimentation, result traceability, and clear technical communication for research and engineering review.

The codebase contains final training implementations, selected notebooks, benchmark evidence, and smoke-test validation paths suitable for practical reproducibility.

## Methodology

The benchmark combines:
- PPO as a strong model-free baseline
- MBPO-style model-based training iterations (including feature-only and JAX variants)
- Shared game environment dynamics and aligned reward instrumentation
- Metadata-backed result reporting for objective final comparison

## Benchmark Summary

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

## Final Notebook Set

The notebook set is curated to the strongest and most presentation-ready artifacts:
- notebooks/flappy_bird_mbpo.ipynb: final MBPO benchmark source notebook
- notebooks/flappy_bird_ppo.ipynb: PPO benchmark/baseline notebook
- notebooks/flappy_bird_comparison.ipynb: comparative analysis notebook

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

## Final Result Bundle

The results/ directory contains the publishable benchmark evidence:
- benchmark_summary.json: consolidated comparison values
- best_mbpo_metadata.json: final MBPO metrics snapshot
- ppo_reference_metadata.json: final PPO reference metrics
- fig4_combined_panel.png: benchmark figure used in the README

## Notes for Recruiters

This project demonstrates:
- Practical deep RL engineering (PPO, world models, vectorized environments)
- Iterative experimentation across multiple MBPO versions
- Reproducibility discipline via checkpoints, metrics, and smoke tests
- Cross-framework work in PyTorch and JAX

## Attribution

The game implementation under flappy_bird_repo/ is adapted from an external Flappy Bird RL codebase included in this workspace, with its own LICENSE retained.

## License

This project is licensed under the MIT License. See LICENSE.
