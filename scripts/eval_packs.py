"""
Evaluate production mistake packs end-to-end.

Run:
    python scripts/eval_packs.py --episodes 20 --difficulty hard
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def heuristic_action(obs) -> AgentAction:
    if obs.security_alert:
        return AgentAction(action_type=ActionType.REFUSE_INJECTION.value, reasoning="Security alert; refusing injection")
    if obs.compliance_warnings:
        return AgentAction(
            action_type=ActionType.CHECK_COMPLIANCE.value,
            compliance_check_result="violation",
            compliance_alternative="safe_alternative",
            decision_documented="Documented compliance risk and mitigation.",
            reasoning="Compliance warning present; checking policy",
        )
    if obs.consecutive_failures >= 2:
        return AgentAction(action_type=ActionType.RECOVER.value, recovery_strategy="retry_with_backoff", reasoning="Recovering")
    if obs.current_step > 3 and obs.budget_remaining > 0.15:
        return AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result="Completed with validation and policy compliance.",
            reasoning="Task appears stable and complete.",
        )
    tool = obs.available_tools[obs.current_step % len(obs.available_tools)]
    return AgentAction(
        action_type=ActionType.CALL_TOOL.value,
        tool_name=tool,
        idempotency_key=f"{obs.episode_id}-{obs.current_step}-{tool}",
        reasoning="Executing next planned tool call with idempotency.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = AgentGauntletEnvironment(seed=args.seed, adaptive_curriculum=False)
    all_pack_metrics: Dict[str, List[float]] = defaultdict(list)

    for ep in range(args.episodes):
        obs = env.reset(difficulty=args.difficulty, seed=args.seed + ep)
        for _ in range(obs.max_steps):
            act = heuristic_action(obs)
            obs = env.step(act)
            if obs.is_done:
                break
        pack_metrics = obs.metadata.get("pack_metrics", {})
        for pack_name, vals in pack_metrics.items():
            for metric_name, metric_val in vals.items():
                all_pack_metrics[f"{pack_name}.{metric_name}"].append(float(metric_val))

    summary = {
        k: (sum(v) / len(v) if v else 0.0)
        for k, v in sorted(all_pack_metrics.items())
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
