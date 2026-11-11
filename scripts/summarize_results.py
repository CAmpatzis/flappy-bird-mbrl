import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "results"
    mbpo = load_json(base / "best_mbpo_metadata.json")
    ppo = load_json(base / "ppo_reference_metadata.json")

    mbpo_avg = float(mbpo["average_reward_last_100"])
    mbpo_max = float(mbpo["max_reward"])
    ppo_mean = float(ppo["mean_score"])
    ppo_max = float(ppo["max_score"])

    print("Benchmark Summary")
    print("=================")
    print(f"MBPO avg reward (last 100): {mbpo_avg:.3f}")
    print(f"MBPO max reward:            {mbpo_max:.3f}")
    print(f"PPO mean score:             {ppo_mean:.3f}")
    print(f"PPO max score:              {ppo_max:.3f}")
    print("-")
    print(f"Avg delta (MBPO - PPO):     {mbpo_avg - ppo_mean:+.3f}")
    print(f"Max delta (MBPO - PPO):     {mbpo_max - ppo_max:+.3f}")


if __name__ == "__main__":
    main()
