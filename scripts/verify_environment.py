"""
Agent Gauntlet — Environment Verifier

Run this BEFORE training to confirm:
1. reset() works
2. step() works
3. rewards are sensible (not all zero, not all same)
4. timeouts work
5. failure injection works
6. multi-agent messaging works
7. reward hacking is hard (trivial completion penalized)

Usage:
    python scripts/verify_environment.py --url http://localhost:8000
    python scripts/verify_environment.py --url https://your-space.hf.space
"""

from __future__ import annotations

import argparse
import sys

from agent_gauntlet import AgentAction, AgentGauntletEnv
from agent_gauntlet.models import ActionType, DifficultyLevel


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return condition


def run_verification(url: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Agent Gauntlet — Environment Verification")
    print(f"URL: {url}")
    print(f"{'='*60}\n")

    all_passed = True

    with AgentGauntletEnv(base_url=url).sync() as env:

        # ----------------------------------------------------------------
        # Test 1: reset() works
        # ----------------------------------------------------------------
        print("Test 1: reset()")
        result = env.reset()
        obs = result.observation
        all_passed &= check("reset() returns observation", obs is not None)
        all_passed &= check("task_description not empty", bool(obs.task_description))
        all_passed &= check("available_tools not empty", len(obs.available_tools) > 0)
        all_passed &= check("budget_remaining = 1.0", obs.budget_remaining == 1.0,
                            f"got {obs.budget_remaining}")
        all_passed &= check("current_step = 0", obs.current_step == 0)
        all_passed &= check("is_done = False", not obs.is_done)
        print()

        # ----------------------------------------------------------------
        # Test 2: step() works
        # ----------------------------------------------------------------
        print("Test 2: step() basic")
        result = env.step(AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[0],
            reasoning="Verifier test step",
        ))
        obs2 = result.observation
        all_passed &= check("step() returns observation", obs2 is not None)
        all_passed &= check("step increments", obs2.current_step == 1,
                            f"got {obs2.current_step}")
        all_passed &= check("reward is float", isinstance(result.reward, float),
                            f"got {type(result.reward)}")
        all_passed &= check("reward in [-1, 1]", -1.0 <= result.reward <= 1.0,
                            f"got {result.reward}")
        print()

        # ----------------------------------------------------------------
        # Test 3: budget tracking works (FIX verification)
        # ----------------------------------------------------------------
        print("Test 3: budget tracking")
        env.reset()
        r1 = env.step(AgentAction(action_type=ActionType.CALL_TOOL.value,
                                   tool_name=obs.available_tools[0], reasoning="test"))
        r2 = env.step(AgentAction(action_type=ActionType.CALL_TOOL.value,
                                   tool_name=obs.available_tools[0], reasoning="test"))
        all_passed &= check("budget decreases after tool calls",
                            r2.observation.budget_remaining < 1.0,
                            f"got {r2.observation.budget_remaining}")
        all_passed &= check("budget_remaining consistent",
                            r2.observation.budget_remaining < r1.observation.budget_remaining,
                            f"r1={r1.observation.budget_remaining}, r2={r2.observation.budget_remaining}")
        print()

        # ----------------------------------------------------------------
        # Test 4: anti-gaming — trivial completion penalized
        # ----------------------------------------------------------------
        print("Test 4: anti-gaming (trivial completion)")
        env.reset()
        result = env.step(AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result="done",
            reasoning="trivial",
        ))
        all_passed &= check("trivial completion gives low/negative reward",
                            result.reward <= 0.1,
                            f"got {result.reward}")
        print()

        # ----------------------------------------------------------------
        # Test 5: failure detection rewarded
        # ----------------------------------------------------------------
        print("Test 5: failure detection")
        # Run until we hit a failure (try multiple episodes)
        failure_found = False
        for attempt in range(5):
            env.reset()
            for step in range(20):
                r = env.step(AgentAction(
                    action_type=ActionType.CALL_TOOL.value,
                    tool_name=obs.available_tools[0],
                    reasoning="looking for failure",
                ))
                if r.observation.last_tool_result and not r.observation.last_tool_result.success:
                    failure_type = r.observation.last_tool_result.failure_type
                    # Now detect it
                    r_detect = env.step(AgentAction(
                        action_type=ActionType.DETECT_FAILURE.value,
                        failure_detected=failure_type,
                        reasoning="detected failure",
                    ))
                    all_passed &= check("correct detection gives positive reward",
                                        r_detect.reward > 0,
                                        f"got {r_detect.reward}")
                    failure_found = True
                    break
            if failure_found:
                break
        if not failure_found:
            print("  ⚠️  SKIP  No failure encountered in 5 episodes (may be normal for easy)")
        print()

        # ----------------------------------------------------------------
        # Test 6: multi-agent messaging
        # ----------------------------------------------------------------
        print("Test 6: multi-agent messaging")
        env.reset(domain="multi_agent_coordination")
        r = env.step(AgentAction(
            action_type=ActionType.SEND_MESSAGE.value,
            target_agent_id="agent_1",
            message_content="Please start processing subtask A",
            reasoning="delegating work",
        ))
        all_passed &= check("send_message doesn't crash", r.observation is not None)
        all_passed &= check("other_agents_status populated",
                            len(r.observation.other_agents_status) > 0,
                            f"got {r.observation.other_agents_status}")
        print()

        # ----------------------------------------------------------------
        # Test 7: reward variance (not all same value)
        # ----------------------------------------------------------------
        print("Test 7: reward variance")
        env.reset()
        rewards = []
        for _ in range(8):
            r = env.step(AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name=obs.available_tools[0],
                reasoning="variance test",
            ))
            rewards.append(r.reward)
        reward_range = max(rewards) - min(rewards)
        all_passed &= check("rewards have variance (not all same)",
                            reward_range > 0.01,
                            f"range={reward_range:.4f}, values={[round(r,3) for r in rewards]}")
        print()

        # ----------------------------------------------------------------
        # Test 8: episode terminates
        # ----------------------------------------------------------------
        print("Test 8: episode termination")
        env.reset()
        done = False
        for _ in range(200):
            r = env.step(AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name=obs.available_tools[0],
                reasoning="termination test",
            ))
            if r.done:
                done = True
                break
        all_passed &= check("episode terminates within max_steps", done,
                            f"termination_reason={r.observation.termination_reason}")
        print()

        # ----------------------------------------------------------------
        # Test 9: active_policies exposed from reset() (Fix #2)
        # ----------------------------------------------------------------
        print("Test 9: active_policies in reset() observation")
        result9 = env.reset(difficulty="medium")
        obs9 = result9.observation
        all_passed &= check("active_policies populated at reset",
                            len(obs9.active_policies) > 0,
                            f"got {obs9.active_policies}")
        all_passed &= check("sla_limit_ms set at reset",
                            obs9.sla_limit_ms > 0,
                            f"got {obs9.sla_limit_ms}")
        print()

        # ----------------------------------------------------------------
        # Test 10: security breach detection rewarded
        # ----------------------------------------------------------------
        print("Test 10: security breach handling")
        security_found = False
        for attempt in range(10):
            env.reset(difficulty="hard")
            for _ in range(30):
                r = env.step(AgentAction(
                    action_type=ActionType.CALL_TOOL.value,
                    tool_name=obs.available_tools[0],
                    reasoning="looking for security breach",
                ))
                if getattr(r.observation, "security_alert", None):
                    # Refuse the injection
                    r_refuse = env.step(AgentAction(
                        action_type=ActionType.REFUSE_INJECTION.value,
                        injection_refused=True,
                        injection_description="detected injection",
                        reasoning="refusing injected instruction",
                    ))
                    all_passed &= check("refuse_injection gives positive reward",
                                        r_refuse.reward > 0,
                                        f"got {r_refuse.reward:.4f}")
                    security_found = True
                    break
            if security_found:
                break
        if not security_found:
            print("  ⚠️  SKIP  No security breach in 10 hard episodes (try expert difficulty)")
        print()

        # ----------------------------------------------------------------
        # Test 11: compliance violation detection rewarded
        # ----------------------------------------------------------------
        print("Test 11: compliance violation handling")
        compliance_found = False
        for attempt in range(10):
            env.reset(difficulty="hard")
            for _ in range(30):
                r = env.step(AgentAction(
                    action_type=ActionType.CALL_TOOL.value,
                    tool_name=obs.available_tools[0],
                    reasoning="looking for compliance violation",
                ))
                if getattr(r.observation, "compliance_warnings", []):
                    policy = r.observation.active_policies[0] if r.observation.active_policies else "UNKNOWN"
                    r_comp = env.step(AgentAction(
                        action_type=ActionType.CHECK_COMPLIANCE.value,
                        compliance_check_result="violation",
                        compliance_policy=policy,
                        compliance_alternative="use_compliant_alternative",
                        decision_documented="documented decision",
                        reasoning="compliance violation detected",
                    ))
                    all_passed &= check("check_compliance gives positive reward",
                                        r_comp.reward > 0,
                                        f"got {r_comp.reward:.4f}")
                    compliance_found = True
                    break
            if compliance_found:
                break
        if not compliance_found:
            print("  ⚠️  SKIP  No compliance violation in 10 hard episodes")
        print()

        # ----------------------------------------------------------------
        # Test 12: SLA tracking works
        # ----------------------------------------------------------------
        print("Test 12: SLA tracking")
        env.reset(difficulty="hard")
        sla_tracked = False
        for _ in range(20):
            r = env.step(AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name=obs.available_tools[0],
                reasoning="sla test",
            ))
            if getattr(r.observation, "last_step_latency_ms", 0) > 0:
                sla_tracked = True
                all_passed &= check("last_step_latency_ms populated",
                                    r.observation.last_step_latency_ms > 0,
                                    f"got {r.observation.last_step_latency_ms:.0f}ms")
                all_passed &= check("sla_limit_ms populated",
                                    r.observation.sla_limit_ms > 0,
                                    f"got {r.observation.sla_limit_ms:.0f}ms")
                break
        if not sla_tracked:
            print("  ⚠️  SKIP  No tool call returned latency")
        print()

        # ----------------------------------------------------------------
        # Test 13: observability trace rewarded
        # ----------------------------------------------------------------
        print("Test 13: observability trace quality")
        env.reset()
        r_trace = env.step(AgentAction(
            action_type=ActionType.GENERATE_TRACE.value,
            diagnostic_trace=(
                "Step 3 failed because rate limit hit after 3 rapid calls. "
                "Root cause: no delay between calls. "
                "Next time: add 1s delay after every 2nd call."
            ),
            reasoning="generating trace after failure",
        ))
        all_passed &= check("generate_trace returns valid reward",
                            isinstance(r_trace.reward, float),
                            f"got {r_trace.reward:.4f}")
        all_passed &= check("episode_traces populated",
                            len(getattr(r_trace.observation, "episode_traces", [])) > 0,
                            f"got {getattr(r_trace.observation, 'episode_traces', [])}")
        print()

        # ----------------------------------------------------------------
        # Test 14: checkpoint_state saves and last_checkpoint_step updates (Fix #4)
        # ----------------------------------------------------------------
        print("Test 14: checkpoint_state / last_checkpoint_step")
        env.reset()
        # Take a few steps first
        for _ in range(3):
            env.step(AgentAction(action_type=ActionType.CALL_TOOL.value,
                                  tool_name=obs.available_tools[0], reasoning="pre-checkpoint"))
        r_ckpt = env.step(AgentAction(
            action_type=ActionType.CHECKPOINT_STATE.value,
            checkpoint_data='{"completed": ["step1", "step2"], "step": 3, "pending": ["step4"]}',
            reasoning="checkpointing state",
        ))
        all_passed &= check("checkpoint_state returns valid reward",
                            isinstance(r_ckpt.reward, float),
                            f"got {r_ckpt.reward:.4f}")
        all_passed &= check("context_checkpoint_available=True after checkpoint",
                            getattr(r_ckpt.observation, "context_checkpoint_available", False),
                            f"got {getattr(r_ckpt.observation, 'context_checkpoint_available', False)}")
        all_passed &= check("last_checkpoint_step > 0 after checkpoint",
                            getattr(r_ckpt.observation, "last_checkpoint_step", 0) > 0,
                            f"got {getattr(r_ckpt.observation, 'last_checkpoint_step', 0)}")
        print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"{'='*60}")
    if all_passed:
        print("✅ ALL CHECKS PASSED — environment is ready for training")
    else:
        print("❌ SOME CHECKS FAILED — fix issues before training")
    print(f"{'='*60}\n")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    passed = run_verification(args.url)
    sys.exit(0 if passed else 1)

