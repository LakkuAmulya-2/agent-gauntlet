"""
Agent Gauntlet — Before/After Demo Script

Guide Section 19: "baseline model attempt → reward/verifier output → trained model attempt → measurable improvement"

"""
# ============================================================
# GENERATE_RESULTS: standalone results generator
# Run: python scripts/demo_before_after.py --generate-results
# Produces real reward curves from real environment episodes.
# ============================================================

import argparse as _argparse
import sys as _sys

_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--generate-results", action="store_true")
_args, _remaining = _parser.parse_known_args()

if _args.generate_results:
    import json, pathlib, random, time
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from agent_gauntlet.runtime.environment import AgentGauntletEnvironment
    from agent_gauntlet.models import (
        DifficultyLevel, AgentAction, ActionType, FailureType
    )
    from agent_gauntlet.runtime.scenarios import RECOVERY_STRATEGIES

    pathlib.Path("assets").mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # Policy definitions
    # ----------------------------------------------------------------

    def random_policy(obs, _state):
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=random.choice(obs.available_tools),
            reasoning="Random baseline",
        ), _state

    def smart_policy(obs, state):
        """Heuristic policy that detects and recovers from failures."""
        last_failure = state.get("last_failure")
        step = state.get("step", 0)
        state["step"] = step + 1

        # Security breach
        if getattr(obs, "security_alert", None):
            state["last_failure"] = None
            return AgentAction(
                action_type=ActionType.REFUSE_INJECTION.value,
                injection_refused=True,
                injection_description=obs.security_alert[:80],
                reasoning="Security alert — refusing injected instruction",
            ), state

        # Compliance warning
        if getattr(obs, "compliance_warnings", []):
            policy = obs.active_policies[0] if obs.active_policies else "UNKNOWN"
            state["last_failure"] = None
            return AgentAction(
                action_type=ActionType.CHECK_COMPLIANCE.value,
                compliance_check_result="violation",
                compliance_policy=policy,
                compliance_alternative="use_compliant_alternative",
                decision_documented=f"Policy {policy} violation — using compliant alternative",
                reasoning=f"Compliance warning for {policy}",
            ), state

        # Context pressure — checkpoint
        if obs.context_used_pct > 0.75 and not getattr(obs, "context_checkpoint_available", False):
            return AgentAction(
                action_type=ActionType.CHECKPOINT_STATE.value,
                checkpoint_data=f'{{"completed": {len(obs.completed_checkpoints)}, "step": {obs.current_step}}}',
                reasoning="Context >75% — checkpointing state",
            ), state

        # Detect failure
        if obs.last_tool_result and not obs.last_tool_result.success:
            ft = obs.last_tool_result.failure_type
            if ft != "none" and last_failure != ft:
                state["last_failure"] = ft
                return AgentAction(
                    action_type=ActionType.DETECT_FAILURE.value,
                    failure_detected=ft,
                    reasoning=f"Tool failed HTTP {obs.last_tool_result.status_code}",
                ), state

        # Recover
        if last_failure:
            ft_enum = next((f for f in FailureType if f.value == last_failure), FailureType.API_500)
            recovery = RECOVERY_STRATEGIES.get(ft_enum, "retry_with_backoff")
            state["last_failure"] = None
            return AgentAction(
                action_type=ActionType.RECOVER.value,
                recovery_strategy=recovery,
                reasoning=f"Applying {recovery}",
            ), state

        # Generate trace after failure
        if obs.consecutive_failures > 0:
            return AgentAction(
                action_type=ActionType.GENERATE_TRACE.value,
                diagnostic_trace=(
                    f"Step {obs.current_step} failed because of consecutive failures. "
                    "Root cause: upstream service instability. "
                    "Next time: add exponential backoff between retries."
                ),
                reasoning="Generating diagnostic trace after failure",
            ), state

        # Escalate on 3+ consecutive failures
        if obs.consecutive_failures >= 3:
            return AgentAction(
                action_type=ActionType.ESCALATE.value,
                escalation_reason="3+ consecutive failures",
                reasoning="Cannot recover automatically",
            ), state

        # Complete near end
        if obs.current_step >= obs.max_steps - 3 and obs.consecutive_failures == 0:
            return AgentAction(
                action_type=ActionType.COMPLETE_TASK.value,
                task_result=f"Completed {len(obs.completed_checkpoints)} of {len(obs.completed_checkpoints) + len(obs.pending_objectives)} objectives",
                reasoning="Near end of episode",
            ), state

        # Default: call next tool
        tool_idx = step % len(obs.available_tools)
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[tool_idx],
            reasoning=f"Executing step {step}",
        ), state

    # ----------------------------------------------------------------
    # Episode runner
    # ----------------------------------------------------------------

    def run_episodes(policy_fn, n_episodes, difficulty, seed_offset=0):
        results = []
        for ep in range(n_episodes):
            env = AgentGauntletEnvironment(
                seed=seed_offset + ep,
                adaptive_curriculum=False,
            )
            obs = env.reset(difficulty=difficulty)
            ep_reward = 0.0
            step_rewards = []
            state = {}
            while not obs.is_done:
                action, state = policy_fn(obs, state)
                obs = env.step(action)
                r = getattr(obs, "_reward", 0.0)
                ep_reward += r
                step_rewards.append(r)
            meta = obs.metadata or {}
            results.append({
                "episode": ep,
                "total_reward": ep_reward,
                "step_rewards": step_rewards,
                "steps": obs.current_step,
                "completed": obs.termination_reason == "task_completed",
                "failures_detected": meta.get("failures_detected_correctly", 0),
                "total_failures": meta.get("total_injected_failures", 0),
                "successful_recoveries": meta.get("successful_recoveries", 0),
                "recovery_attempts": meta.get("recovery_attempts", 1),
                "budget_remaining": obs.budget_remaining,
                "security_refused": meta.get("injections_refused", 0),
                "compliance_detected": meta.get("compliance_violations_detected", 0),
                "sla_breaches": meta.get("sla_breaches", 0),
                "traces_generated": meta.get("diagnostic_traces_count", 0),
            })
            print(f"  ep {ep+1:2d}: reward={ep_reward:+.4f} steps={obs.current_step:3d} "
                  f"done={obs.termination_reason} "
                  f"det={meta.get('failures_detected_correctly',0)}/{meta.get('total_injected_failures',0)}")
        return results

    # ----------------------------------------------------------------
    # Run both policies
    # ----------------------------------------------------------------

    N_EPISODES = 50
    DIFFICULTY = "easy"

    print(f"\n{'='*60}")
    print(f"Agent Gauntlet — Real Results Generation")
    print(f"Episodes: {N_EPISODES} | Difficulty: {DIFFICULTY}")
    print(f"{'='*60}\n")

    print("Running RANDOM baseline...")
    random.seed(0)
    random_results = run_episodes(random_policy, N_EPISODES, DIFFICULTY, seed_offset=0)

    print("\nRunning SMART (heuristic) policy...")
    smart_results = run_episodes(smart_policy, N_EPISODES, DIFFICULTY, seed_offset=1000)

    # ----------------------------------------------------------------
    # Compute summary metrics
    # ----------------------------------------------------------------

    def summarize(results):
        n = len(results)
        total_rewards = [r["total_reward"] for r in results]
        return {
            "avg_reward": sum(total_rewards) / n,
            "std_reward": float(np.std(total_rewards)),
            "task_completion_rate": sum(r["completed"] for r in results) / n,
            "failure_detection_rate": (
                sum(r["failures_detected"] for r in results) /
                max(1, sum(r["total_failures"] for r in results))
            ),
            "recovery_rate": (
                sum(r["successful_recoveries"] for r in results) /
                max(1, sum(r["recovery_attempts"] for r in results))
            ),
            "avg_budget_remaining": sum(r["budget_remaining"] for r in results) / n,
            "avg_steps": sum(r["steps"] for r in results) / n,
            "security_refusal_rate": (
                sum(r["security_refused"] for r in results) / max(1, n)
            ),
            "compliance_detection_rate": (
                sum(r["compliance_detected"] for r in results) / max(1, n)
            ),
            "avg_traces": sum(r["traces_generated"] for r in results) / n,
        }

    rand_summary = summarize(random_results)
    smart_summary = summarize(smart_results)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Random':>10} {'Smart':>10}")
    print(f"{'-'*55}")
    metrics = [
        ("Avg episode reward", "avg_reward", ".4f"),
        ("Task completion rate", "task_completion_rate", ".1%"),
        ("Failure detection rate", "failure_detection_rate", ".1%"),
        ("Recovery rate", "recovery_rate", ".1%"),
        ("Avg budget remaining", "avg_budget_remaining", ".2f"),
        ("Avg steps", "avg_steps", ".1f"),
        ("Security refusal rate", "security_refusal_rate", ".2f"),
        ("Compliance detection rate", "compliance_detection_rate", ".2f"),
        ("Avg diagnostic traces", "avg_traces", ".2f"),
    ]
    for label, key, fmt in metrics:
        rv = format(rand_summary[key], fmt)
        sv = format(smart_summary[key], fmt)
        print(f"  {label:<33} {rv:>10} {sv:>10}")

    # ----------------------------------------------------------------
    # Plot 1: Reward curves (rolling average)
    # ----------------------------------------------------------------

    rand_rewards = [r["total_reward"] for r in random_results]
    smart_rewards = [r["total_reward"] for r in smart_results]
    episodes = list(range(1, N_EPISODES + 1))

    window = 10
    def rolling(vals, w):
        return [np.mean(vals[max(0,i-w):i+1]) for i in range(len(vals))]

    rand_smooth = rolling(rand_rewards, window)
    smart_smooth = rolling(smart_rewards, window)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: reward curves
    axes[0].plot(episodes, rand_rewards, alpha=0.25, color="gray")
    axes[0].plot(episodes, rand_smooth, color="gray", linewidth=2, label=f"Random baseline (avg={rand_summary['avg_reward']:.3f})")
    axes[0].plot(episodes, smart_rewards, alpha=0.25, color="steelblue")
    axes[0].plot(episodes, smart_smooth, color="steelblue", linewidth=2, label=f"Smart heuristic (avg={smart_summary['avg_reward']:.3f})")
    axes[0].axhline(y=rand_summary["avg_reward"], color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Agent Gauntlet — Episode Reward\n(Random vs Smart Heuristic, easy difficulty)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: per-metric comparison bar chart
    bar_metrics = [
        ("Task\ncompletion", "task_completion_rate"),
        ("Failure\ndetection", "failure_detection_rate"),
        ("Recovery\nrate", "recovery_rate"),
        ("Budget\nremaining", "avg_budget_remaining"),
    ]
    x = np.arange(len(bar_metrics))
    width = 0.35
    rand_vals = [rand_summary[k] for _, k in bar_metrics]
    smart_vals = [smart_summary[k] for _, k in bar_metrics]
    axes[1].bar(x - width/2, rand_vals, width, label="Random baseline", color="gray", alpha=0.8)
    axes[1].bar(x + width/2, smart_vals, width, label="Smart heuristic", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m for m, _ in bar_metrics])
    axes[1].set_ylabel("Rate / Score")
    axes[1].set_title("Per-Metric Comparison\n(Random vs Smart Heuristic)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("assets/reward_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: assets/reward_curves.png")

    # ----------------------------------------------------------------
    # Plot 2: Per-component reward breakdown
    # ----------------------------------------------------------------

    # Collect step-level rewards per episode for component analysis
    # We use the smart policy results and break down by episode outcome
    completed_rewards = [r["total_reward"] for r in smart_results if r["completed"]]
    failed_rewards = [r["total_reward"] for r in smart_results if not r["completed"]]
    detected_rewards = [r["total_reward"] for r in smart_results if r["failures_detected"] > 0]
    missed_rewards = [r["total_reward"] for r in smart_results if r["failures_detected"] == 0 and r["total_failures"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: reward distribution by outcome
    data_groups = []
    labels_groups = []
    if completed_rewards:
        data_groups.append(completed_rewards)
        labels_groups.append(f"Task completed\n(n={len(completed_rewards)})")
    if failed_rewards:
        data_groups.append(failed_rewards)
        labels_groups.append(f"Task failed\n(n={len(failed_rewards)})")
    if detected_rewards:
        data_groups.append(detected_rewards)
        labels_groups.append(f"Failures detected\n(n={len(detected_rewards)})")
    if missed_rewards:
        data_groups.append(missed_rewards)
        labels_groups.append(f"Failures missed\n(n={len(missed_rewards)})")

    if data_groups:
        bp = axes[0].boxplot(data_groups, labels=labels_groups, patch_artist=True)
        colors = ["steelblue", "tomato", "mediumseagreen", "orange"]
        for patch, color in zip(bp["boxes"], colors[:len(data_groups)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward Distribution by Episode Outcome\n(Smart Heuristic, easy difficulty)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: capability metrics comparison
    cap_labels = ["Security\nRefusal", "Compliance\nDetection", "Avg\nTraces", "Budget\nEfficiency"]
    rand_cap = [
        rand_summary["security_refusal_rate"],
        rand_summary["compliance_detection_rate"],
        min(1.0, rand_summary["avg_traces"] / 3.0),
        rand_summary["avg_budget_remaining"],
    ]
    smart_cap = [
        smart_summary["security_refusal_rate"],
        smart_summary["compliance_detection_rate"],
        min(1.0, smart_summary["avg_traces"] / 3.0),
        smart_summary["avg_budget_remaining"],
    ]
    x2 = np.arange(len(cap_labels))
    axes[1].bar(x2 - width/2, rand_cap, width, label="Random baseline", color="gray", alpha=0.8)
    axes[1].bar(x2 + width/2, smart_cap, width, label="Smart heuristic", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cap_labels)
    axes[1].set_ylabel("Rate / Normalized score")
    axes[1].set_title("New Capability Metrics\n(Security · Compliance · Observability · Efficiency)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("assets/component_rewards.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: assets/component_rewards.png")

    # ----------------------------------------------------------------
    # Save raw results JSON for README
    # ----------------------------------------------------------------

    results_data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "n_episodes": N_EPISODES,
        "difficulty": DIFFICULTY,
        "random_baseline": rand_summary,
        "smart_heuristic": smart_summary,
        "improvement": {
            k: smart_summary[k] - rand_summary[k]
            for k in rand_summary
        }
    }
    pathlib.Path("assets/results.json").write_text(
        json.dumps(results_data, indent=2), encoding="utf-8"
    )
    print("Saved: assets/results.json")

    print(f"\n{'='*60}")
    print("DONE — plots saved to assets/")
    print(f"  reward_curves.png   — episode reward comparison")
    print(f"  component_rewards.png — capability metrics breakdown")
    print(f"  results.json        — raw numbers for README")
    print(f"{'='*60}")
    _sys.exit(0)

Generates the demo comparison judges want to see:
1. Untrained model on a task with failures
2. Trained model on the same task
3. Side-by-side metrics

Usage:
    python scripts/demo_before_after.py \
        --trained-model outputs/gauntlet-easy-20260425 \
        --url http://localhost:8000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from agent_gauntlet import AgentAction, AgentGauntletEnv
from agent_gauntlet.models import ActionType


def run_episode_with_model(
    model_dir: str,
    env_url: str,
    seed: int,
    difficulty: str,
    label: str,
) -> Dict:
    """Run one episode with a model and collect metrics."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("transformers not installed")
        return {}

    from train_grpo import SYSTEM_PROMPT

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    metrics = {
        "label": label,
        "total_reward": 0.0,
        "task_completed": False,
        "failures_detected": 0,
        "failures_missed": 0,
        "valid_json_rate": 0.0,
        "steps_taken": 0,
        "budget_remaining": 1.0,
        "actions": [],
    }

    valid_json_count = 0
    total_steps = 0

    with AgentGauntletEnv(base_url=env_url).sync() as env:
        result = env.reset(difficulty=difficulty, seed=seed)
        obs = result.observation

        print(f"\n[{label}] Task: {obs.task_description[:80]}...")

        for step in range(obs.max_steps):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"TASK: {obs.task_description}\n"
                    f"TOOLS: {', '.join(obs.available_tools)}\n"
                    f"Step {obs.current_step}/{obs.max_steps} | "
                    f"Budget: {obs.api_calls_made}/{obs.api_calls_budget} | "
                    f"Context: {obs.context_used_pct:.0%}"
                    + (f"\nLast error: {obs.recent_errors[-1]}" if obs.recent_errors else "")
                )}
            ]

            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=False, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=200,
                    temperature=0.1, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            total_steps += 1
            try:
                action_data = json.loads(generated)
                valid_json_count += 1
                action = AgentAction(
                    action_type=action_data.get("action_type", "call_tool"),
                    tool_name=action_data.get("tool_name"),
                    reasoning=action_data.get("reasoning", ""),
                    failure_detected=action_data.get("failure_detected"),
                    recovery_strategy=action_data.get("recovery_strategy"),
                    task_result=action_data.get("task_result"),
                )
            except json.JSONDecodeError:
                action = AgentAction(
                    action_type=ActionType.CALL_TOOL.value,
                    tool_name=obs.available_tools[0] if obs.available_tools else "unknown",
                    reasoning="fallback",
                )

            result = env.step(action)
            metrics["total_reward"] += result.reward
            metrics["actions"].append(action.action_type)

            # Track failure handling
            if obs.last_tool_result and not obs.last_tool_result.success:
                if action.failure_detected:
                    metrics["failures_detected"] += 1
                else:
                    metrics["failures_missed"] += 1

            obs = result.observation
            if obs.is_done:
                metrics["task_completed"] = obs.termination_reason == "task_completed"
                metrics["budget_remaining"] = obs.budget_remaining
                break

    metrics["steps_taken"] = total_steps
    metrics["valid_json_rate"] = valid_json_count / max(1, total_steps)
    return metrics


def print_comparison(before: Dict, after: Dict):
    """Print side-by-side comparison for demo."""
    print(f"\n{'='*70}")
    print(f"BEFORE vs AFTER - Agent Gauntlet Demo")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Before (untrained)':<22} {'After (trained)':<22}")
    print(f"{'-'*70}")

    metrics = [
        ("Total reward", "total_reward", ".4f"),
        ("Task completed", "task_completed", ""),
        ("Failures detected", "failures_detected", "d"),
        ("Failures missed", "failures_missed", "d"),
        ("Valid JSON rate", "valid_json_rate", ".1%"),
        ("Steps taken", "steps_taken", "d"),
        ("Budget remaining", "budget_remaining", ".2f"),
    ]

    for label, key, fmt in metrics:
        b_val = before.get(key, "N/A")
        a_val = after.get(key, "N/A")
        if fmt and b_val != "N/A":
            b_str = format(b_val, fmt)
            a_str = format(a_val, fmt)
        else:
            b_str = str(b_val)
            a_str = str(a_val)

        # Add improvement indicator
        improved = ""
        if key in ("total_reward", "failures_detected", "valid_json_rate", "budget_remaining"):
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                improved = " ↑" if a_val > b_val else (" ↓" if a_val < b_val else " =")
        elif key in ("failures_missed",):
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                improved = " ↓" if a_val < b_val else (" ↑" if a_val > b_val else " =")

        print(f"{label:<30} {b_str:<22} {a_str + improved:<22}")

    print(f"{'='*70}")

    # Action sequence comparison
    print(f"\nAction sequences:")
    print(f"  Before: {' → '.join(before.get('actions', [])[:8])}")
    print(f"  After:  {' → '.join(after.get('actions', [])[:8])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B",
                        help="Untrained base model for comparison")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42,
                        help="Same seed = same task for fair comparison")
    parser.add_argument("--difficulty", default="easy")
    args = parser.parse_args()

    print("Running before/after comparison...")
    print(f"Seed: {args.seed} (same task for both models)")

    before = run_episode_with_model(
        args.base_model, args.url, args.seed, args.difficulty, "Untrained"
    )
    after = run_episode_with_model(
        args.trained_model, args.url, args.seed, args.difficulty, "Trained"
    )

    print_comparison(before, after)

    # Save for README
    import json as _json
    with open("demo_results.json", "w") as f:
        _json.dump({"before": before, "after": after}, f, indent=2)
    print("\nSaved to demo_results.json — add numbers to README")


if __name__ == "__main__":
    main()
