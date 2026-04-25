"""
Judge-readiness preflight for OpenEnv Hackathon submission.

Usage:
    python scripts/judge_readiness.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _exists(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def check_manifest(results: list[tuple[str, bool, str]]) -> None:
    path = REPO_ROOT / "openenv.yaml"
    ok = path.exists()
    results.append(("openenv.yaml present", ok, "Required OpenEnv manifest"))
    if not ok:
        return
    text = path.read_text(encoding="utf-8")
    keys_ok = all(k in text for k in ["spec_version", "name:", "tools:", "tasks:"])
    results.append(("openenv.yaml has required sections", keys_ok, "spec_version/name/tools/tasks"))


def check_training_assets(results: list[tuple[str, bool, str]]) -> None:
    results.append(("train_grpo.py exists", _exists("train_grpo.py"), "TRL/GRPO training script"))
    results.append(("train_sft.py exists", _exists("train_sft.py"), "SFT warmup script"))
    results.append(
        ("Colab notebook exists", _exists("notebooks/agent_gauntlet_grpo.ipynb"), "Judge-runnable notebook")
    )
    results.append(("results JSON exists", _exists("assets/results.json"), "Before/after metrics artifact"))
    results.append(("reward plot exists", _exists("assets/reward_curves.png"), "Training evidence"))
    results.append(("component plot exists", _exists("assets/component_rewards.png"), "Reward evidence"))


def check_docs(results: list[tuple[str, bool, str]]) -> None:
    readme_ok = _exists("README.md")
    blog_ok = _exists("blog_post.md")
    results.append(("README exists", readme_ok, "Submission must include README"))
    results.append(("Blog draft exists", blog_ok, "Submission must include writeup/video"))
    if not readme_ok:
        return

    readme = _read("README.md")
    has_space_link = "HuggingFace Space" in readme
    results.append(("README includes HF Space section", has_space_link, "Discoverable environment URL"))

    no_placeholder = all(
        marker not in readme
        for marker in [
            "add final",
            "add link",
            "your-space.hf.space",
            "YOUR_USERNAME",
        ]
    )
    results.append(("README has no obvious placeholder links", no_placeholder, "Judge-facing materials should be final"))


def check_results_integrity(results: list[tuple[str, bool, str]]) -> None:
    path = REPO_ROOT / "assets/results.json"
    if not path.exists():
        results.append(("results.json parseable", False, "Missing metrics file"))
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = [
            "n_episodes",
            "difficulty",
            "random_baseline",
            "smart_heuristic",
        ]
        has_fields = all(field in data for field in fields)
        results.append(("results.json parseable", True, "Valid JSON"))
        results.append(("results.json has required sections", has_fields, "n_episodes/difficulty/baselines"))
    except json.JSONDecodeError:
        results.append(("results.json parseable", False, "Invalid JSON"))


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    check_manifest(checks)
    check_training_assets(checks)
    check_docs(checks)
    check_results_integrity(checks)

    width = max(len(name) for name, _, _ in checks) + 2
    failed = 0
    print("OpenEnv Hackathon Judge Readiness")
    print("=" * 44)
    for name, ok, note in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{name:<{width}} {status}  - {note}")
    print("-" * 44)
    print(f"Total: {len(checks)} checks | Failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
