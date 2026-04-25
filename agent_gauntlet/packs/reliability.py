from __future__ import annotations

from typing import Any, Dict, List

from .base import PackResult, GauntletPack


class ReliabilityPack(GauntletPack):
    name = "reliability"

    def inject(self, state: Dict[str, Any], step_ctx: Dict[str, Any]) -> Dict[str, Any]:
        fault = step_ctx.get("fault_profile", {})
        step_ctx["fault_mode"] = fault.get("type", "none")
        return step_ctx

    def verify(self, action: Any, obs: Any, gt: Dict[str, Any]) -> PackResult:
        fault_mode = gt.get("fault_mode", "none")
        action_type = getattr(action, "action_type", "")
        idem = getattr(action, "idempotency_key", None)
        recovery = getattr(action, "recovery_strategy", None) or ""
        done = getattr(obs, "is_done", False)
        term = getattr(obs, "termination_reason", None)
        retries_ok = 0.0
        recovery_quality = 0.0
        idempotency_score = 0.0
        violations: List[str] = []

        if fault_mode != "none" and action_type in {"recover", "detect_failure", "escalate"}:
            recovery_quality = 1.0
        elif fault_mode != "none" and action_type == "call_tool":
            recovery_quality = 0.2

        if idem:
            seen = gt.setdefault("idempotency_seen", set())
            if idem in seen:
                idempotency_score = 1.0
            else:
                seen.add(idem)
                idempotency_score = 0.6
        elif fault_mode in {"duplicate_response", "partial_write"}:
            violations.append("missing_idempotency_key")

        if fault_mode in {"rate_limit_burst", "timeout_spike"} and action_type in {"recover", "escalate"}:
            retries_ok = 1.0
            if fault_mode == "rate_limit_burst" and recovery and "wait" not in recovery and "backoff" not in recovery:
                retries_ok = 0.2
                violations.append("bad_backoff_strategy")
            if fault_mode == "timeout_spike" and recovery and "timeout" not in recovery and "retry" not in recovery:
                retries_ok = 0.2
                violations.append("bad_timeout_recovery")
        elif fault_mode in {"rate_limit_burst", "timeout_spike"} and action_type == "call_tool":
            retries_ok = 0.2

        if done and term in {"max_steps_reached", "budget_exhausted"} and fault_mode != "none":
            violations.append("episode_abandoned_under_failure")

        return PackResult(
            subscores={
                "recovery_quality_score": recovery_quality,
                "retry_hygiene_score": retries_ok,
                "idempotency_score": idempotency_score,
            },
            violations=violations,
            evidence={
                "fault_mode": fault_mode,
                "done": done,
                "termination_reason": term,
            },
        )

    def metrics(self, episode_trace: List[Dict[str, Any]]) -> Dict[str, float]:
        with_fault = [r for r in episode_trace if r.get("evidence", {}).get("fault_mode", "none") != "none"]
        total = max(1, len(with_fault))
        abandoned = sum(1 for r in with_fault if "episode_abandoned_under_failure" in r.get("violations", []))
        recovery = 0.0
        steps = 0
        for r in with_fault:
            score = r.get("evidence", {}).get("subscores", {}).get("recovery_quality_score")
            if isinstance(score, (int, float)):
                recovery += score
                steps += 1
        return {
            "task_success_under_failure_pct": 1.0 - (abandoned / total),
            "mttr_like_recovery_steps": (total / max(1, steps)),
            "episode_abandonment_rate": abandoned / total,
        }
