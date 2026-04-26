"""
Update README trained-evidence section from generated assets.

Reads:
  - assets/trained_vs_random.json
  - assets/training_summary_latest.json (optional)

Updates a marked block in README:
  <!-- TRAINED_EVIDENCE_START -->
  ...
  <!-- TRAINED_EVIDENCE_END -->
"""

from __future__ import annotations

import json
from pathlib import Path


START = "<!-- TRAINED_EVIDENCE_START -->"
END = "<!-- TRAINED_EVIDENCE_END -->"


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("README markers for trained evidence block are missing.")

    trained = _load_json(Path("assets/trained_vs_random.json")) or {}
    summary = _load_json(Path("assets/training_summary_latest.json")) or {}

    rb = trained.get("random_baseline", {})
    tm = trained.get("trained_model", {})
    delta = trained.get("delta_avg_reward")

    rows = [
        "| Metric | Random Baseline | Trained Model |",
        "|---|---|---|",
        f"| Avg episode reward | {rb.get('avg_reward', 'n/a')} | {tm.get('avg_reward', 'n/a')} |",
        f"| Task completion rate | {rb.get('task_completion_rate', 'n/a')} | {tm.get('task_completion_rate', 'n/a')} |",
        f"| Failure detection rate | {rb.get('failure_detection_rate', 'n/a')} | {tm.get('failure_detection_rate', 'n/a')} |",
        f"| Recovery rate | {rb.get('recovery_rate', 'n/a')} | {tm.get('recovery_rate', 'n/a')} |",
        f"| Avg budget remaining | {rb.get('avg_budget_remaining', 'n/a')} | {tm.get('avg_budget_remaining', 'n/a')} |",
    ]
    delta_line = f"**Delta avg reward:** `{delta}`" if delta is not None else "**Delta avg reward:** `n/a`"

    training_line = (
        f"Training summary: planned updates `{summary.get('planned_update_steps', 'n/a')}`, "
        f"reward points `{summary.get('reward_points', 'n/a')}`, "
        f"last10 reward mean `{summary.get('reward_last10_mean', 'n/a')}`."
    )

    block = "\n".join(
        [
            START,
            "### Trained vs Random (Auto-generated)",
            "",
            delta_line,
            "",
            *rows,
            "",
            training_line,
            "",
            "Artifacts: `assets/trained_vs_random.json`, `assets/trained_vs_random.png`, `assets/training_summary_latest.json`",
            END,
        ]
    )

    pre = text.split(START)[0]
    post = text.split(END)[1]
    readme.write_text(pre + block + post, encoding="utf-8")
    print("README trained evidence block updated.")


if __name__ == "__main__":
    main()
