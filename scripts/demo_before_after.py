"""
Agent Gauntlet - Before/After metrics generator.

Runs random vs heuristic policies against a running environment server and writes:
- assets/results.json
- assets/reward_curves.png
- assets/component_rewards.png
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

from agent_gauntlet import AgentAction, AgentGauntletEnv
from agent_gauntlet.models import ActionType, FailureType
from agent_gauntlet.runtime.scenarios import RECOVERY_STRATEGIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class EpisodeMetrics:
    reward: float
    completed: bool
    detected: int
    total_failures: int
    recoveries: int
    recovery_attempts: int
    budget: float
    steps: int
    security_refused: int
    compliance_detected: int
    traces: int


def random_policy(obs, state):
    action = AgentAction(
        action_type=ActionType.CALL_TOOL.value,
        tool_name=random.choice(obs.available_tools),
        reasoning="Random baseline policy",
    )
    return action, state


def smart_policy(obs, state):
    step = state.get("step", 0)
    state["step"] = step + 1
    last_failure = state.get("last_failure")

    if getattr(obs, "security_alert", None):
        state["last_failure"] = None
        return AgentAction(
            action_type=ActionType.REFUSE_INJECTION.value,
            injection_refused=True,
            injection_description=obs.security_alert[:120],
            reasoning="Security alert detected, refusing instruction",
        ), state

    if getattr(obs, "compliance_warnings", []):
        policy = obs.active_policies[0] if obs.active_policies else "UNKNOWN"
        state["last_failure"] = None
        return AgentAction(
            action_type=ActionType.CHECK_COMPLIANCE.value,
            compliance_check_result="violation",
            compliance_policy=policy,
            compliance_alternative="use_compliant_alternative",
            decision_documented=f"Policy {policy} violation detected",
            reasoning="Check policy and choose compliant path",
        ), state

    if obs.context_used_pct > 0.75 and not getattr(obs, "context_checkpoint_available", False):
        return AgentAction(
            action_type=ActionType.CHECKPOINT_STATE.value,
            checkpoint_data=json.dumps({"step": obs.current_step, "pending": len(obs.pending_objectives)}),
            reasoning="Context pressure detected, checkpointing state",
        ), state

    if obs.last_tool_result and not obs.last_tool_result.success:
        failure = obs.last_tool_result.failure_type
        if failure != FailureType.NONE.value and last_failure != failure:
            state["last_failure"] = failure
            return AgentAction(
                action_type=ActionType.DETECT_FAILURE.value,
                failure_detected=failure,
                reasoning="Tool call failed, classifying failure type",
            ), state

    if last_failure:
        failure_enum = next((f for f in FailureType if f.value == last_failure), FailureType.API_500)
        recovery = RECOVERY_STRATEGIES.get(failure_enum, "retry_with_backoff")
        state["last_failure"] = None
        return AgentAction(
            action_type=ActionType.RECOVER.value,
            recovery_strategy=recovery,
            reasoning=f"Applying {recovery}",
        ), state

    if obs.consecutive_failures > 0:
        return AgentAction(
            action_type=ActionType.GENERATE_TRACE.value,
            diagnostic_trace="Failure observed. Root cause likely upstream instability. Add retry/backoff.",
            reasoning="Generate trace after failure",
        ), state

    if obs.consecutive_failures >= 3:
        return AgentAction(
            action_type=ActionType.ESCALATE.value,
            escalation_reason="Three consecutive failures",
            reasoning="Escalating after repeated failures",
        ), state

    if obs.current_step >= obs.max_steps - 3 and obs.consecutive_failures == 0:
        return AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result=f"Completed {len(obs.completed_checkpoints)} objectives",
            reasoning="Near episode end with stable progress",
        ), state

    tool_idx = step % len(obs.available_tools)
    return AgentAction(
        action_type=ActionType.CALL_TOOL.value,
        tool_name=obs.available_tools[tool_idx],
        reasoning=f"Progress step {step}",
    ), state


def run_episodes(policy_fn, n_episodes: int, difficulty: str, env_url: str, seed_offset: int = 0):
    results = []
    for ep in range(n_episodes):
        random.seed(seed_offset + ep)
        with AgentGauntletEnv(base_url=env_url).sync() as env:
            reset_result = env.reset(difficulty=difficulty)
            obs = reset_result.observation
            ep_reward = 0.0
            state = {}

            while not obs.is_done:
                action, state = policy_fn(obs, state)
                step_result = env.step(action)
                ep_reward += step_result.reward
                obs = step_result.observation

        meta = obs.metadata or {}
        detected = meta.get("failures_detected_correctly", 0)
        total_failures = meta.get("total_injected_failures", 0)
        results.append(
            EpisodeMetrics(
                reward=ep_reward,
                completed=obs.termination_reason == "task_completed",
                detected=detected,
                total_failures=total_failures,
                recoveries=meta.get("successful_recoveries", 0),
                recovery_attempts=max(1, meta.get("recovery_attempts", 1)),
                budget=obs.budget_remaining,
                steps=obs.current_step,
                security_refused=meta.get("injections_refused", 0),
                compliance_detected=meta.get("compliance_violations_detected", 0),
                traces=meta.get("diagnostic_traces_count", 0),
            )
        )
    return results


def summarize(results: list[EpisodeMetrics]):
    n = len(results)
    rewards = [r.reward for r in results]
    return {
        "avg_reward": sum(rewards) / n,
        "std_reward": float(np.std(rewards)),
        "task_completion_rate": sum(r.completed for r in results) / n,
        "failure_detection_rate": sum(r.detected for r in results) / max(1, sum(r.total_failures for r in results)),
        "recovery_rate": sum(r.recoveries for r in results) / max(1, sum(r.recovery_attempts for r in results)),
        "avg_budget_remaining": sum(r.budget for r in results) / n,
        "avg_steps": sum(r.steps for r in results) / n,
        "security_refusal_rate": sum(r.security_refused for r in results) / n,
        "compliance_detection_rate": sum(r.compliance_detected for r in results) / n,
        "avg_traces": sum(r.traces for r in results) / n,
        "all_rewards": rewards,
    }


def _rolling(values: list[float], window: int = 10):
    return [float(np.mean(values[max(0, i - window): i + 1])) for i in range(len(values))]


def save_plots(random_summary: dict, smart_summary: dict, smart_results: list[EpisodeMetrics], n_episodes: int):
    Path("assets").mkdir(exist_ok=True)
    episodes = list(range(1, n_episodes + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(episodes, random_summary["all_rewards"], alpha=0.2, color="gray")
    axes[0].plot(episodes, _rolling(random_summary["all_rewards"]), color="gray", lw=2.2, label="Random")
    axes[0].plot(episodes, smart_summary["all_rewards"], alpha=0.2, color="steelblue")
    axes[0].plot(episodes, _rolling(smart_summary["all_rewards"]), color="steelblue", lw=2.2, label="Heuristic")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Episode Reward Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    names = ["Task completion", "Failure detection", "Recovery", "Budget remaining"]
    keys = ["task_completion_rate", "failure_detection_rate", "recovery_rate", "avg_budget_remaining"]
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width / 2, [random_summary[k] for k in keys], width, label="Random", color="gray", alpha=0.8)
    axes[1].bar(x + width / 2, [smart_summary[k] for k in keys], width, label="Heuristic", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Core Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("assets/reward_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    completed_rewards = [r.reward for r in smart_results if r.completed]
    failed_rewards = [r.reward for r in smart_results if not r.completed]
    if completed_rewards or failed_rewards:
        plot_data = []
        labels = []
        if completed_rewards:
            plot_data.append(completed_rewards)
            labels.append("Completed")
        if failed_rewards:
            plot_data.append(failed_rewards)
            labels.append("Failed")
        axes[0].boxplot(plot_data, tick_labels=labels, patch_artist=True)
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward by Outcome")
    axes[0].grid(True, alpha=0.3, axis="y")

    cap_names = ["Security refusal", "Compliance detect", "Diagnostic traces", "Budget efficiency"]
    random_cap = [
        random_summary["security_refusal_rate"],
        random_summary["compliance_detection_rate"],
        min(1.0, random_summary["avg_traces"] / 3.0),
        random_summary["avg_budget_remaining"],
    ]
    smart_cap = [
        smart_summary["security_refusal_rate"],
        smart_summary["compliance_detection_rate"],
        min(1.0, smart_summary["avg_traces"] / 3.0),
        smart_summary["avg_budget_remaining"],
    ]
    x2 = np.arange(len(cap_names))
    axes[1].bar(x2 - width / 2, random_cap, width, label="Random", color="gray", alpha=0.8)
    axes[1].bar(x2 + width / 2, smart_cap, width, label="Heuristic", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cap_names)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Extended Capability Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("assets/component_rewards.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate before/after metrics and plots.")
    parser.add_argument("--url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard", "expert"])
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    random_results = run_episodes(random_policy, args.episodes, args.difficulty, args.url, seed_offset=0)
    smart_results = run_episodes(smart_policy, args.episodes, args.difficulty, args.url, seed_offset=10000)

    random_summary = summarize(random_results)
    smart_summary = summarize(smart_results)

    save_plots(random_summary, smart_summary, smart_results, args.episodes)

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_episodes": args.episodes,
        "difficulty": args.difficulty,
        "env_url": args.url,
        "random_baseline": {k: v for k, v in random_summary.items() if k != "all_rewards"},
        "smart_heuristic": {k: v for k, v in smart_summary.items() if k != "all_rewards"},
    }
    Path("assets").mkdir(exist_ok=True)
    Path("assets/results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
