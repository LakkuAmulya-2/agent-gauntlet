"""
Deterministic replay validator for judge reproducibility checks.

Usage:
  python scripts/replay_episode.py --seed 42 --difficulty hard --domain api_workflow --runs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def run_episode(seed: int, difficulty: str, domain: str | None) -> Dict[str, object]:
    env = AgentGauntletEnvironment(seed=seed, adaptive_curriculum=False)
    obs = env.reset(seed=seed, difficulty=difficulty, domain=domain)
    for i in range(min(obs.max_steps, 40)):
        action = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[i % len(obs.available_tools)],
            idempotency_key=f"{obs.episode_id}:{i}",
            confidence_score=0.6,
            reasoning="Deterministic replay probe",
        )
        obs = env.step(action)
        if obs.is_done:
            break
    payload = {
        "termination_reason": obs.termination_reason,
        "reward": getattr(obs, "_reward", 0.0),
        "metadata_subset": {
            "failures_detected_correctly": obs.metadata.get("failures_detected_correctly"),
            "successful_recoveries": obs.metadata.get("successful_recoveries"),
            "sla_breaches": obs.metadata.get("sla_breaches"),
            "coordination_conflicts_detected": obs.metadata.get("coordination_conflicts_detected"),
            "uncertainty_overconfidence_events": obs.metadata.get("uncertainty_overconfidence_events"),
        },
        "pack_metrics": obs.metadata.get("pack_metrics", {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["digest"] = digest
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    domain = args.domain or None
    rows: List[Dict[str, object]] = [
        run_episode(seed=args.seed, difficulty=args.difficulty, domain=domain)
        for _ in range(max(1, args.runs))
    ]
    digests = [r["digest"] for r in rows]
    stable = len(set(digests)) == 1
    print(json.dumps({"stable": stable, "digests": digests, "sample": rows[0]}, indent=2))
    if not stable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
