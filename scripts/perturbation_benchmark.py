"""
Sim-to-prod perturbation benchmark report generator.

Outputs JSON summary for judge evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def run(seed: int, difficulty: str, domain: str | None) -> Dict[str, float]:
    env = AgentGauntletEnvironment(seed=seed, adaptive_curriculum=False)
    obs = env.reset(seed=seed, difficulty=difficulty, domain=domain)
    reward_sum = 0.0
    for i in range(min(obs.max_steps, 45)):
        action = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[i % len(obs.available_tools)],
            idempotency_key=f"{obs.episode_id}:{i}",
            confidence_score=0.9 if i % 7 == 0 else 0.55,
            reasoning="Perturbation benchmark run",
        )
        res = env.step(action)
        obs = res
        reward_sum += float(getattr(obs, "_reward", 0.0))
        if obs.is_done:
            break
    pack = obs.metadata.get("pack_metrics", {})
    scal = pack.get("scalability", {})
    rel = pack.get("reliability", {})
    hall = pack.get("hallucination", {})
    return {
        "reward": reward_sum,
        "sla_compliance_under_load": float(scal.get("sla_compliance_under_load", 0.0)),
        "throughput": float(scal.get("throughput", 0.0)),
        "task_success_under_failure_pct": float(rel.get("task_success_under_failure_pct", 0.0)),
        "hallucination_rate": float(hall.get("hallucination_rate", 0.0)),
        "overconfidence_events": float(obs.metadata.get("uncertainty_overconfidence_events", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--out", type=str, default="assets/perturbation_report.json")
    args = parser.parse_args()

    domain = args.domain or None
    agg: Dict[str, List[float]] = defaultdict(list)
    for i in range(max(1, args.episodes)):
        row = run(seed=args.seed + i, difficulty=args.difficulty, domain=domain)
        for k, v in row.items():
            agg[k].append(v)
    summary = {k: mean(v) for k, v in agg.items()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
