"""
Concurrent stress runner for scalability pack.

Run:
    python scripts/stress_load.py --workers 10 --episodes 30
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from typing import Dict, List

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def run_episode(seed: int, difficulty: str) -> Dict[str, float]:
    env = AgentGauntletEnvironment(seed=seed, adaptive_curriculum=False)
    obs = env.reset(difficulty=difficulty, seed=seed)
    latencies: List[float] = []
    for i in range(obs.max_steps):
        tool = obs.available_tools[i % len(obs.available_tools)]
        obs = env.step(
            AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name=tool,
                idempotency_key=f"{obs.episode_id}:{i}:{tool}",
                reasoning="Load test step",
            )
        )
        latencies.append(float(obs.last_step_latency_ms or 0.0))
        if obs.is_done:
            break
    return {
        "steps": float(len(latencies)),
        "p50": median(latencies) if latencies else 0.0,
        "p95": sorted(latencies)[int(0.95 * max(0, len(latencies) - 1))] if latencies else 0.0,
        "p99": sorted(latencies)[int(0.99 * max(0, len(latencies) - 1))] if latencies else 0.0,
        "sla_breaches": float(obs.sla_breaches),
        "reward": float(getattr(obs, "_reward", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    rows: List[Dict[str, float]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_episode, args.seed + i, args.difficulty) for i in range(args.episodes)]
        for f in as_completed(futures):
            rows.append(f.result())

    n = max(1, len(rows))
    throughput = sum(r["steps"] for r in rows) / n
    sla_compliance = 1.0 - (sum(1 for r in rows if r["sla_breaches"] > 0) / n)
    avg_reward = sum(r["reward"] for r in rows) / n

    print("Stress summary")
    print(f"episodes={n} workers={args.workers}")
    print(f"throughput_steps_per_episode={throughput:.2f}")
    print(f"sla_compliance_under_load={sla_compliance:.3f}")
    print(f"cost_reward_efficiency_at_scale={avg_reward:.3f}")


if __name__ == "__main__":
    main()
