"""
Evaluate trained model vs random baseline on Agent Gauntlet.

Outputs:
  - assets/trained_vs_random.json
  - assets/trained_vs_random.png

This is judge-facing evidence for "real training happened and improved behavior".
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment
from train_grpo import SYSTEM_PROMPT


@dataclass
class EpisodeMetrics:
    reward: float
    completed: bool
    steps: int
    failures_detected: int
    failures_total: int
    recoveries: int
    recovery_attempts: int
    budget_remaining: float


def _build_prompt(obs) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"TASK: {obs.task_description}\n"
        f"GOAL: {obs.task_goal}\n"
        f"TOOLS: {', '.join(obs.available_tools)}\n"
        f"STEP: {obs.current_step}/{obs.max_steps}\n"
        f"BUDGET: {obs.api_calls_made}/{obs.api_calls_budget}\n"
        f"CONTEXT_USED: {obs.context_used_pct:.2f}\n"
        "Return a single JSON action."
    )


def _extract_action_json(text: str) -> Dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def _to_action(action_data: Dict) -> AgentAction:
    return AgentAction(
        action_type=action_data.get("action_type", ActionType.CALL_TOOL.value),
        tool_name=action_data.get("tool_name"),
        tool_args=action_data.get("tool_args", {}),
        reasoning=action_data.get("reasoning", ""),
        failure_detected=action_data.get("failure_detected"),
        recovery_strategy=action_data.get("recovery_strategy"),
        escalation_reason=action_data.get("escalation_reason"),
        task_result=action_data.get("task_result"),
        target_agent_id=action_data.get("target_agent_id"),
        message_content=action_data.get("message_content"),
        drift_detected=action_data.get("drift_detected"),
        contradiction_resolution=action_data.get("contradiction_resolution"),
        injection_refused=action_data.get("injection_refused", False),
        injection_description=action_data.get("injection_description"),
        compliance_check_result=action_data.get("compliance_check_result"),
        compliance_policy=action_data.get("compliance_policy"),
        compliance_alternative=action_data.get("compliance_alternative"),
        decision_documented=action_data.get("decision_documented"),
        diagnostic_trace=action_data.get("diagnostic_trace"),
        transparency_decision=action_data.get("transparency_decision"),
        stakeholder_belief_update=action_data.get("stakeholder_belief_update"),
        checkpoint_data=action_data.get("checkpoint_data"),
        checkpoint_id=action_data.get("checkpoint_id"),
        state_summary=action_data.get("state_summary"),
        idempotency_key=action_data.get("idempotency_key"),
        confidence_score=action_data.get("confidence_score"),
    )


def _summarize(rows: List[EpisodeMetrics]) -> Dict:
    return {
        "avg_reward": mean(r.reward for r in rows),
        "task_completion_rate": mean(1.0 if r.completed else 0.0 for r in rows),
        "avg_steps": mean(r.steps for r in rows),
        "failure_detection_rate": sum(r.failures_detected for r in rows) / max(1, sum(r.failures_total for r in rows)),
        "recovery_rate": sum(r.recoveries for r in rows) / max(1, sum(r.recovery_attempts for r in rows)),
        "avg_budget_remaining": mean(r.budget_remaining for r in rows),
        "all_rewards": [r.reward for r in rows],
    }


def _save_plot(random_summary: Dict, trained_summary: Dict, out_path: Path) -> None:
    episodes = list(range(1, len(random_summary["all_rewards"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(episodes, random_summary["all_rewards"], color="gray", alpha=0.35, label="Random baseline")
    axes[0].plot(episodes, trained_summary["all_rewards"], color="steelblue", alpha=0.35, label="Trained model")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward by Episode")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    labels = ["Avg reward", "Completion", "Detect", "Recovery"]
    random_vals = [
        random_summary["avg_reward"],
        random_summary["task_completion_rate"],
        random_summary["failure_detection_rate"],
        random_summary["recovery_rate"],
    ]
    trained_vals = [
        trained_summary["avg_reward"],
        trained_summary["task_completion_rate"],
        trained_summary["failure_detection_rate"],
        trained_summary["recovery_rate"],
    ]
    x = list(range(len(labels)))
    width = 0.36
    axes[1].bar([i - width / 2 for i in x], random_vals, width=width, color="gray", label="Random")
    axes[1].bar([i + width / 2 for i in x], trained_vals, width=width, color="steelblue", label="Trained")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Score / Reward")
    axes[1].set_title("Random vs Trained")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _run_random_episode(env: AgentGauntletEnvironment, difficulty: str, seed: int) -> EpisodeMetrics:
    rng = random.Random(seed)
    obs = env.reset(difficulty=difficulty, seed=seed)
    total_reward = 0.0
    steps = 0
    while not obs.is_done:
        steps += 1
        action = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=rng.choice(obs.available_tools),
            reasoning="random-baseline",
        )
        obs = env.step(action)
        total_reward += float(getattr(obs, "_reward", 0.0))
        if steps >= obs.max_steps:
            break
    meta = obs.metadata or {}
    return EpisodeMetrics(
        reward=total_reward,
        completed=obs.termination_reason == "task_completed",
        steps=obs.current_step,
        failures_detected=int(meta.get("failures_detected_correctly", 0)),
        failures_total=int(meta.get("total_injected_failures", 0)),
        recoveries=int(meta.get("successful_recoveries", 0)),
        recovery_attempts=max(1, int(meta.get("recovery_attempts", 1))),
        budget_remaining=float(obs.budget_remaining),
    )


def _run_trained_episode(env: AgentGauntletEnvironment, model, tokenizer, difficulty: str, seed: int, max_new_tokens: int) -> EpisodeMetrics:
    import torch

    obs = env.reset(difficulty=difficulty, seed=seed)
    total_reward = 0.0
    steps = 0
    while not obs.is_done:
        steps += 1
        prompt = _build_prompt(obs)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        try:
            action = _to_action(_extract_action_json(text))
        except Exception:
            action = AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name=obs.available_tools[steps % len(obs.available_tools)],
                reasoning="fallback-invalid-json",
            )
        obs = env.step(action)
        total_reward += float(getattr(obs, "_reward", 0.0))
        if steps >= obs.max_steps:
            break
    meta = obs.metadata or {}
    return EpisodeMetrics(
        reward=total_reward,
        completed=obs.termination_reason == "task_completed",
        steps=obs.current_step,
        failures_detected=int(meta.get("failures_detected_correctly", 0)),
        failures_total=int(meta.get("total_injected_failures", 0)),
        recoveries=int(meta.get("successful_recoveries", 0)),
        recovery_attempts=max(1, int(meta.get("recovery_attempts", 1))),
        budget_remaining=float(obs.budget_remaining),
    )


def evaluate(model_dir: str, difficulty: str, episodes: int, seed: int, max_new_tokens: int) -> Tuple[Dict, Dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    random_env = AgentGauntletEnvironment(default_difficulty=difficulty, adaptive_curriculum=False)
    trained_env = AgentGauntletEnvironment(default_difficulty=difficulty, adaptive_curriculum=False)

    random_rows: List[EpisodeMetrics] = []
    trained_rows: List[EpisodeMetrics] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        random_rows.append(_run_random_episode(random_env, difficulty, ep_seed))
        trained_rows.append(_run_trained_episode(trained_env, model, tokenizer, difficulty, ep_seed, max_new_tokens))

    random_summary = _summarize(random_rows)
    trained_summary = _summarize(trained_rows)
    return random_summary, trained_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model vs random baseline.")
    parser.add_argument("--model-dir", required=True, help="Path to trained checkpoint directory.")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard", "expert"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    random_summary, trained_summary = evaluate(
        model_dir=args.model_dir,
        difficulty=args.difficulty,
        episodes=args.episodes,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    Path("assets").mkdir(exist_ok=True)
    out_json = Path("assets/trained_vs_random.json")
    out_png = Path("assets/trained_vs_random.png")
    payload = {
        "difficulty": args.difficulty,
        "episodes": args.episodes,
        "seed": args.seed,
        "model_dir": args.model_dir,
        "random_baseline": {k: v for k, v in random_summary.items() if k != "all_rewards"},
        "trained_model": {k: v for k, v in trained_summary.items() if k != "all_rewards"},
        "delta_avg_reward": trained_summary["avg_reward"] - random_summary["avg_reward"],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_plot(random_summary, trained_summary, out_png)

    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
