"""
Agent Gauntlet — Baseline Evaluation Script

Run BEFORE training to establish baseline metrics.
Run AFTER training to compare improvement.

FAQ Q56: "debug order: environment → verifier → scripted baseline → frozen model → tiny RL"
This script covers the "scripted baseline" step.

Usage:
    python scripts/run_baseline.py --url http://localhost:8000 --episodes 20
    python scripts/run_baseline.py --url http://localhost:8000 --policy smart
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List

from agent_gauntlet import AgentAction, AgentGauntletEnv
from agent_gauntlet.models import ActionType, FailureType

# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Pure random — establishes floor."""
    name = "random"

    def act(self, obs) -> AgentAction:
        tool = random.choice(obs.available_tools)
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=tool,
            reasoning="random",
        )


class AlwaysCompletePolicy:
    """Immediately completes — tests anti-gaming."""
    name = "always_complete"

    def act(self, obs) -> AgentAction:
        return AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result="done",
            reasoning="immediate completion",
        )


class SmartPolicy:
    """
    Heuristic policy — detects failures and recovers.
    Handles all new capabilities: security, compliance, SLA, checkpointing.
    Establishes a reasonable upper bound for untrained models.
    """
    name = "smart_heuristic"

    FAILURE_TO_RECOVERY = {
        "api_500": "retry_with_backoff",
        "rate_limit_429": "wait_and_retry",
        "auth_401": "refresh_token_and_retry",
        "malformed_response": "use_fallback_parser",
        "timeout": "retry_with_timeout_increase",
        "adversarial_input": "sanitize_and_retry",
        "context_pressure": "summarize_state",
        "cascading": "rollback_to_last_checkpoint",
        # Fix #9: new failure types
        "security_breach": "refuse_and_escalate",
        "compliance_violation": "find_compliant_alternative",
        "sla_breach": "optimize_and_continue",
    }

    def __init__(self):
        self._last_failure = None
        self._step = 0

    def act(self, obs) -> AgentAction:
        self._step += 1

        # Fix #9: handle security breach first (highest priority)
        if getattr(obs, "security_alert", None):
            return AgentAction(
                action_type=ActionType.REFUSE_INJECTION.value,
                injection_refused=True,
                injection_description=obs.security_alert[:100],
                reasoning="Security alert — refusing injected instruction and escalating",
            )

        # Fix #9: handle compliance violation
        if getattr(obs, "compliance_warnings", []):
            policy = obs.active_policies[0] if obs.active_policies else "UNKNOWN_POLICY"
            return AgentAction(
                action_type=ActionType.CHECK_COMPLIANCE.value,
                compliance_check_result="violation",
                compliance_policy=policy,
                compliance_alternative="use_compliant_alternative",
                decision_documented=f"Policy {policy} violation — using compliant alternative",
                reasoning=f"Compliance warning for {policy}",
            )

        # Fix #9: checkpoint when context pressure and no checkpoint yet
        if obs.context_used_pct > 0.75 and not getattr(obs, "context_checkpoint_available", False):
            return AgentAction(
                action_type=ActionType.CHECKPOINT_STATE.value,
                checkpoint_data=f'{{"completed": {len(obs.completed_checkpoints)}, "step": {obs.current_step}}}',
                reasoning="Context >75% — checkpointing state",
            )

        # Summarize state if context pressure high
        if obs.context_used_pct > 0.85:
            return AgentAction(
                action_type=ActionType.SUMMARIZE_STATE.value,
                state_summary=f"Completed {len(obs.completed_checkpoints)} objectives. "
                              f"Pending: {obs.pending_objectives[:3]}. "
                              f"Step {obs.current_step}/{obs.max_steps}.",
                reasoning="context pressure high",
            )

        # Detect failure from last tool result
        if obs.last_tool_result and not obs.last_tool_result.success:
            failure_type = obs.last_tool_result.failure_type
            if failure_type != "none" and self._last_failure != failure_type:
                self._last_failure = failure_type
                return AgentAction(
                    action_type=ActionType.DETECT_FAILURE.value,
                    failure_detected=failure_type,
                    reasoning=f"Tool failed with {obs.last_tool_result.status_code}",
                )

        # Recover from detected failure
        if self._last_failure and self._last_failure in self.FAILURE_TO_RECOVERY:
            recovery = self.FAILURE_TO_RECOVERY[self._last_failure]
            self._last_failure = None
            return AgentAction(
                action_type=ActionType.RECOVER.value,
                recovery_strategy=recovery,
                reasoning=f"Applying {recovery}",
            )

        # Escalate if too many consecutive failures
        if obs.consecutive_failures >= 3:
            return AgentAction(
                action_type=ActionType.ESCALATE.value,
                escalation_reason="3+ consecutive failures",
                reasoning="Cannot recover automatically",
            )

        # Complete if near end and no failures
        if obs.current_step >= obs.max_steps - 3 and obs.consecutive_failures == 0:
            return AgentAction(
                action_type=ActionType.COMPLETE_TASK.value,
                task_result=f"Completed {len(obs.completed_checkpoints)} of {len(obs.pending_objectives) + len(obs.completed_checkpoints)} objectives",
                reasoning="Near end of episode",
            )

        # Default: call next tool in sequence
        tool_idx = self._step % len(obs.available_tools)
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[tool_idx],
            reasoning=f"Executing step {self._step}",
        )

    def reset(self):
        self._last_failure = None
        self._step = 0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_policy(env_url: str, policy, n_episodes: int, difficulty: str) -> Dict:
    metrics = defaultdict(list)

    with AgentGauntletEnv(base_url=env_url).sync() as env:
        for ep in range(n_episodes):
            if hasattr(policy, "reset"):
                policy.reset()

            result = env.reset(difficulty=difficulty)
            obs = result.observation
            ep_reward = 0.0
            ep_steps = 0

            while not obs.is_done:
                action = policy.act(obs)
                result = env.step(action)
                ep_reward += result.reward
                obs = result.observation
                ep_steps += 1

            metrics["episode_reward"].append(ep_reward)
            metrics["steps"].append(ep_steps)
            metrics["completed"].append(
                1 if obs.termination_reason == "task_completed" else 0
            )
            metrics["budget_remaining"].append(obs.budget_remaining)

    return {
        "policy": policy.name,
        "difficulty": difficulty,
        "n_episodes": n_episodes,
        "avg_reward": sum(metrics["episode_reward"]) / n_episodes,
        "task_completion_rate": sum(metrics["completed"]) / n_episodes,
        "avg_steps": sum(metrics["steps"]) / n_episodes,
        "avg_budget_remaining": sum(metrics["budget_remaining"]) / n_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--policy", choices=["random", "smart", "all"], default="all")
    args = parser.parse_args()

    policies = []
    if args.policy in ("random", "all"):
        policies.append(RandomPolicy())
    if args.policy in ("smart", "all"):
        policies.append(SmartPolicy())
    if args.policy == "all":
        policies.append(AlwaysCompletePolicy())

    print(f"\nBaseline Evaluation — {args.difficulty} difficulty, {args.episodes} episodes")
    print("=" * 60)

    results = []
    for policy in policies:
        print(f"\nRunning {policy.name}...")
        result = evaluate_policy(args.url, policy, args.episodes, args.difficulty)
        results.append(result)
        print(f"  Avg reward:          {result['avg_reward']:.4f}")
        print(f"  Task completion:     {result['task_completion_rate']:.1%}")
        print(f"  Avg steps:           {result['avg_steps']:.1f}")
        print(f"  Avg budget left:     {result['avg_budget_remaining']:.2f}")

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY (save these numbers — compare after training)")
    print("=" * 60)
    for r in results:
        print(f"{r['policy']:20s}  reward={r['avg_reward']:.4f}  completion={r['task_completion_rate']:.1%}")

    # Save to file for comparison
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to baseline_results.json")


if __name__ == "__main__":
    main()
