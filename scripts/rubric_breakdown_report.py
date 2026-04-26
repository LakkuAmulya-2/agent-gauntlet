"""
Generate a judge-facing rubric breakdown report from latest training summary.

Outputs:
  - assets/rubric_breakdown.json
  - assets/rubric_breakdown.png
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _latest_summary() -> Path:
    candidates = sorted(glob.glob("outputs/gauntlet-*/training_summary.json"))
    if not candidates:
        raise FileNotFoundError("No training_summary.json found under outputs/gauntlet-*/")
    return Path(candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", default="", help="Optional explicit training_summary.json path")
    args = parser.parse_args()

    summary_path = Path(args.summary_path) if args.summary_path else _latest_summary()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rb = summary.get("rubric_breakdown", {})
    if not rb:
        raise RuntimeError("rubric_breakdown missing in training summary.")

    labels = list(rb.keys())
    vals = [rb[k].get("mean") if rb[k].get("mean") is not None else 0.0 for k in labels]

    Path("assets").mkdir(exist_ok=True)
    out_json = Path("assets/rubric_breakdown.json")
    out_png = Path("assets/rubric_breakdown.png")
    out_json.write_text(json.dumps(rb, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 5))
    plt.bar(labels, vals, color="steelblue", alpha=0.85)
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Rubric")
    plt.ylabel("Mean reward contribution")
    plt.title("Composable Rubric Breakdown (from training run)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Loaded: {summary_path}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
