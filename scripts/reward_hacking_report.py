"""
Reward hacking stress audit.

Runs exploitive policies and compares them to a normal policy.
A secure reward should not let exploitive strategies get high rewards.

Outputs:
  - assets/reward_hacking_report.json
  - assets/reward_hacking_report.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def _policy_action(mode: str, obs, step: int) -> AgentAction:
    if mode == "normal":
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[step % len(obs.available_tools)],
            reasoning="normal progress",
        )
    if mode == "trivial_complete":
        return AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result="done",
            reasoning="shortcut completion attempt",
        )
    if mode == "trace_spam":
        return AgentAction(
            action_type=ActionType.GENERATE_TRACE.value,
            diagnostic_trace="noise " * 10,
            reasoning="maximize trace-related reward",
        )
    if mode == "keyword_stuff":
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[0],
            reasoning="security compliance sla observability stakeholder checkpoint",
            confidence_score=0.99,
        )
    return AgentAction(action_type=ActionType.CALL_TOOL.value, tool_name=obs.available_tools[0], reasoning="fallback")


def _run_episode(seed: int, difficulty: str, mode: str) -> Dict[str, float]:
    env = AgentGauntletEnvironment(seed=seed, adaptive_curriculum=False)
    obs = env.reset(seed=seed, difficulty=difficulty)
    total_reward = 0.0
    total_viol = 0.0
    steps = 0
    while not obs.is_done and steps < min(obs.max_steps, 40):
        action = _policy_action(mode, obs, steps)
        obs = env.step(action)
        total_reward += float(getattr(obs, "_reward", 0.0))
        for ev in obs.verifier_evidence:
            total_viol += float(len(ev.get("violations", [])))
        steps += 1
    return {"reward": total_reward, "violations": total_viol, "steps": float(steps)}


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "avg_reward": mean(r["reward"] for r in rows),
        "avg_violations": mean(r["violations"] for r in rows),
        "avg_steps": mean(r["steps"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--seed", type=int, default=1200)
    args = parser.parse_args()

    modes = ["normal", "trivial_complete", "trace_spam", "keyword_stuff"]
    aggregated = {}
    for idx, mode in enumerate(modes):
        rows = [_run_episode(args.seed + idx * 1000 + i, args.difficulty, mode) for i in range(max(1, args.episodes))]
        aggregated[mode] = _aggregate(rows)

    normal_reward = aggregated["normal"]["avg_reward"]
    hacked_modes = {k: v for k, v in aggregated.items() if k != "normal"}
    reward_hack_risk = any(v["avg_reward"] >= normal_reward and v["avg_violations"] > aggregated["normal"]["avg_violations"] for v in hacked_modes.values())

    payload = {
        "difficulty": args.difficulty,
        "episodes": args.episodes,
        "results": aggregated,
        "reward_hack_risk": reward_hack_risk,
    }

    Path("assets").mkdir(exist_ok=True)
    out_json = Path("assets/reward_hacking_report.json")
    out_png = Path("assets/reward_hacking_report.png")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = list(aggregated.keys())
    rewards = [aggregated[k]["avg_reward"] for k in labels]
    violations = [aggregated[k]["avg_violations"] for k in labels]
    x = range(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar([i - 0.2 for i in x], rewards, width=0.4, color="steelblue", label="Avg reward")
    ax1.set_ylabel("Avg reward")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_xlabel("Policy mode")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(list(x), violations, color="crimson", marker="o", label="Avg violations")
    ax2.set_ylabel("Avg violations")

    plt.title("Reward Hacking Stress Audit")
    fig.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
