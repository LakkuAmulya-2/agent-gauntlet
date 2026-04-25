# Agent Gauntlet - Results Generator
"""
Run: python scripts/demo_before_after.py --generate-results
Produces REAL reward curves from REAL environment episodes. No GPU needed.
"""
import argparse as _ap
import sys as _sys

_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--generate-results", action="store_true")
_args, _ = _p.parse_known_args()

if _args.generate_results:
    import json, pathlib, random
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from agent_gauntlet.runtime.environment import AgentGauntletEnvironment
    from agent_gauntlet.models import AgentAction, ActionType, FailureType
    from agent_gauntlet.runtime.scenarios import RECOVERY_STRATEGIES

    pathlib.Path("assets").mkdir(exist_ok=True)

    def random_policy(obs, state):
        return AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=random.choice(obs.available_tools),
            reasoning="Random baseline",
        ), state

    def smart_policy(obs, state):
        step = state.get("step", 0)
        state["step"] = step + 1
        last_failure = state.get("last_failure")

        if getattr(obs, "security_alert", None):
            state["last_failure"] = None
            return AgentAction(action_type=ActionType.REFUSE_INJECTION.value,
                injection_refused=True, injection_description=obs.security_alert[:80],
                reasoning="Security alert - refusing"), state

        if getattr(obs, "compliance_warnings", []):
            policy = obs.active_policies[0] if obs.active_policies else "UNKNOWN"
            state["last_failure"] = None
            return AgentAction(action_type=ActionType.CHECK_COMPLIANCE.value,
                compliance_check_result="violation", compliance_policy=policy,
                compliance_alternative="use_compliant_alternative",
                decision_documented="Policy violation - compliant alternative",
                reasoning="Compliance warning"), state

        if obs.context_used_pct > 0.75 and not getattr(obs, "context_checkpoint_available", False):
            return AgentAction(action_type=ActionType.CHECKPOINT_STATE.value,
                checkpoint_data='{"step":' + str(obs.current_step) + '}',
                reasoning="Context >75% - checkpointing"), state

        if obs.last_tool_result and not obs.last_tool_result.success:
            ft = obs.last_tool_result.failure_type
            if ft != "none" and last_failure != ft:
                state["last_failure"] = ft
                return AgentAction(action_type=ActionType.DETECT_FAILURE.value,
                    failure_detected=ft, reasoning="Tool failed"), state

        if last_failure:
            ft_enum = next((f for f in FailureType if f.value == last_failure), FailureType.API_500)
            recovery = RECOVERY_STRATEGIES.get(ft_enum, "retry_with_backoff")
            state["last_failure"] = None
            return AgentAction(action_type=ActionType.RECOVER.value,
                recovery_strategy=recovery, reasoning="Applying " + recovery), state

        if obs.consecutive_failures > 0:
            return AgentAction(action_type=ActionType.GENERATE_TRACE.value,
                diagnostic_trace="Step " + str(obs.current_step) + " failed. Root cause: upstream instability. Next time: add backoff.",
                reasoning="Generating trace after failure"), state

        if obs.consecutive_failures >= 3:
            return AgentAction(action_type=ActionType.ESCALATE.value,
                escalation_reason="3+ consecutive failures", reasoning="Cannot recover"), state

        if obs.current_step >= obs.max_steps - 3 and obs.consecutive_failures == 0:
            return AgentAction(action_type=ActionType.COMPLETE_TASK.value,
                task_result="Completed " + str(len(obs.completed_checkpoints)) + " objectives",
                reasoning="Near end"), state

        tool_idx = step % len(obs.available_tools)
        return AgentAction(action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[tool_idx], reasoning="Step " + str(step)), state

    def run_episodes(policy_fn, n, difficulty, seed_offset=0):
        results = []
        for ep in range(n):
            env = AgentGauntletEnvironment(seed=seed_offset + ep, adaptive_curriculum=False)
            obs = env.reset(difficulty=difficulty)
            ep_reward, state = 0.0, {}
            while not obs.is_done:
                action, state = policy_fn(obs, state)
                obs = env.step(action)
                ep_reward += getattr(obs, "_reward", 0.0)
            meta = obs.metadata or {}
            det = meta.get("failures_detected_correctly", 0)
            tot = meta.get("total_injected_failures", 0)
            results.append({
                "reward": ep_reward,
                "completed": obs.termination_reason == "task_completed",
                "detected": det, "total_failures": tot,
                "recoveries": meta.get("successful_recoveries", 0),
                "recovery_attempts": max(1, meta.get("recovery_attempts", 1)),
                "budget": obs.budget_remaining, "steps": obs.current_step,
                "security_refused": meta.get("injections_refused", 0),
                "compliance_detected": meta.get("compliance_violations_detected", 0),
                "traces": meta.get("diagnostic_traces_count", 0),
            })
            print("  ep{:2d}: reward={:+.4f}  done={:15s}  det={}/{}".format(
                ep + 1, ep_reward, obs.termination_reason[:15], det, tot))
        return results

    def summarize(results):
        n = len(results)
        rewards = [r["reward"] for r in results]
        return {
            "avg_reward": sum(rewards) / n,
            "std_reward": float(np.std(rewards)),
            "task_completion_rate": sum(r["completed"] for r in results) / n,
            "failure_detection_rate": sum(r["detected"] for r in results) / max(1, sum(r["total_failures"] for r in results)),
            "recovery_rate": sum(r["recoveries"] for r in results) / max(1, sum(r["recovery_attempts"] for r in results)),
            "avg_budget_remaining": sum(r["budget"] for r in results) / n,
            "avg_steps": sum(r["steps"] for r in results) / n,
            "security_refusal_rate": sum(r["security_refused"] for r in results) / max(1, n),
            "compliance_detection_rate": sum(r["compliance_detected"] for r in results) / max(1, n),
            "avg_traces": sum(r["traces"] for r in results) / n,
            "all_rewards": rewards,
        }

    N, DIFF = 50, "easy"
    print("=" * 60)
    print("Agent Gauntlet - Real Results ({} episodes, {})".format(N, DIFF))
    print("=" * 60)

    print("\nRANDOM baseline...")
    random.seed(0)
    rand_r = run_episodes(random_policy, N, DIFF, 0)

    print("\nSMART heuristic...")
    smart_r = run_episodes(smart_policy, N, DIFF, 1000)

    rs, ss = summarize(rand_r), summarize(smart_r)

    print("\n" + "=" * 60)
    for label, key, fmt in [
        ("Avg reward", "avg_reward", ".4f"),
        ("Task completion", "task_completion_rate", ".1%"),
        ("Failure detection", "failure_detection_rate", ".1%"),
        ("Recovery rate", "recovery_rate", ".1%"),
        ("Budget remaining", "avg_budget_remaining", ".2f"),
        ("Security refusal", "security_refusal_rate", ".2f"),
        ("Compliance detect", "compliance_detection_rate", ".2f"),
        ("Avg traces", "avg_traces", ".2f"),
    ]:
        print("  {:<25} {:>10} {:>10}".format(label, format(rs[key], fmt), format(ss[key], fmt)))

    # Plot 1: reward curves
    eps = list(range(1, N + 1))
    w = 10
    roll = lambda v: [float(np.mean(v[max(0,i-w):i+1])) for i in range(len(v))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(eps, rs["all_rewards"], alpha=0.2, color="gray")
    axes[0].plot(eps, roll(rs["all_rewards"]), color="gray", lw=2.5,
                 label="Random (avg={:.3f})".format(rs["avg_reward"]))
    axes[0].plot(eps, ss["all_rewards"], alpha=0.2, color="steelblue")
    axes[0].plot(eps, roll(ss["all_rewards"]), color="steelblue", lw=2.5,
                 label="Smart heuristic (avg={:.3f})".format(ss["avg_reward"]))
    axes[0].axhline(y=rs["avg_reward"], color="gray", ls="--", alpha=0.4)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Agent Gauntlet - Episode Reward\n(Random vs Smart Heuristic, easy)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    bm = [("Task\ncompletion","task_completion_rate"),("Failure\ndetection","failure_detection_rate"),
          ("Recovery\nrate","recovery_rate"),("Budget\nremaining","avg_budget_remaining")]
    x = np.arange(len(bm)); wd = 0.35
    axes[1].bar(x-wd/2, [rs[k] for _,k in bm], wd, label="Random", color="gray", alpha=0.8)
    axes[1].bar(x+wd/2, [ss[k] for _,k in bm], wd, label="Smart", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([m for m,_ in bm])
    axes[1].set_ylabel("Rate / Score")
    axes[1].set_title("Per-Metric Comparison\n(Random vs Smart Heuristic)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y"); axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("assets/reward_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: assets/reward_curves.png")

    # Plot 2: component breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    groups = [(r["reward"] for r in smart_r if r["completed"]),
              (r["reward"] for r in smart_r if not r["completed"]),
              (r["reward"] for r in smart_r if r["detected"] > 0),
              (r["reward"] for r in smart_r if r["detected"] == 0 and r["total_failures"] > 0)]
    labels = ["Completed","Failed","Det. failures","Missed failures"]
    colors = ["steelblue","tomato","mediumseagreen","orange"]
    data_g = [list(g) for g in groups]
    valid = [(d,l,c) for d,l,c in zip(data_g,labels,colors) if d]
    if valid:
        bp = axes[0].boxplot([d for d,_,_ in valid], labels=[l for _,l,_ in valid], patch_artist=True)
        for patch, (_,_,c) in zip(bp["boxes"], valid):
            patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward by Outcome\n(Smart Heuristic, easy)")
    axes[0].grid(True, alpha=0.3, axis="y")

    cl = ["Security\nRefusal","Compliance\nDetect","Diagnostic\nTraces","Budget\nEfficiency"]
    rc = [rs["security_refusal_rate"],rs["compliance_detection_rate"],min(1.0,rs["avg_traces"]/3),rs["avg_budget_remaining"]]
    sc = [ss["security_refusal_rate"],ss["compliance_detection_rate"],min(1.0,ss["avg_traces"]/3),ss["avg_budget_remaining"]]
    x2 = np.arange(len(cl))
    axes[1].bar(x2-wd/2, rc, wd, label="Random", color="gray", alpha=0.8)
    axes[1].bar(x2+wd/2, sc, wd, label="Smart", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(cl)
    axes[1].set_ylabel("Rate / Score")
    axes[1].set_title("New Capability Metrics\n(Security - Compliance - Observability)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y"); axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("assets/component_rewards.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: assets/component_rewards.png")

    out = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "n_episodes": N, "difficulty": DIFF,
        "random_baseline": {k: v for k,v in rs.items() if k != "all_rewards"},
        "smart_heuristic": {k: v for k,v in ss.items() if k != "all_rewards"},
    }
    pathlib.Path("assets/results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved: assets/results.json")
    print("\nDONE - {} real episodes, plots in assets/".format(N))
    _sys.exit(0)

