"""
Multi-seed ablation runner for research-style evidence.

Runs train_grpo with selected reward components disabled and aggregates results.

Outputs:
  - assets/ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_training(seed: int, disable_rewards: str, difficulty: str, dataset_size: int, num_epochs: int, min_update_steps: int) -> str:
    out_dir = f"outputs/ablation-{difficulty}-seed{seed}-disable-{disable_rewards or 'none'}"
    cmd = [
        sys.executable,
        "train_grpo.py",
        "--difficulty",
        difficulty,
        "--dataset-size",
        str(dataset_size),
        "--num-epochs",
        str(num_epochs),
        "--output-dir",
        out_dir,
        "--judge-ready",
        "--min-update-steps",
        str(min_update_steps),
    ]
    if disable_rewards:
        cmd.extend(["--disable-rewards", disable_rewards])
    env = dict(**__import__("os").environ)
    env["PYTHONHASHSEED"] = str(seed)
    subprocess.run(cmd, check=True, env=env)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--dataset-size", type=int, default=1600)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--min-update-steps", type=int, default=80)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    ablations = {
        "full": "",
        "no_security": "security",
        "no_tom": "theory_of_mind",
        "no_observability": "observability",
    }

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    results = {}
    for name, disabled in ablations.items():
        runs = []
        for seed in seed_list:
            out_dir = _run_training(
                seed=seed,
                disable_rewards=disabled,
                difficulty=args.difficulty,
                dataset_size=args.dataset_size,
                num_epochs=args.num_epochs,
                min_update_steps=args.min_update_steps,
            )
            summary_path = Path(out_dir) / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "seed": seed,
                    "output_dir": out_dir,
                    "reward_last10_mean": summary.get("reward_last10_mean"),
                    "reward_mean": summary.get("reward_mean"),
                    "planned_update_steps": summary.get("planned_update_steps"),
                }
            )
        results[name] = runs

    payload = {
        "difficulty": args.difficulty,
        "seeds": seed_list,
        "ablations": results,
    }
    Path("assets").mkdir(exist_ok=True)
    out_path = Path("assets/ablation_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
