from __future__ import annotations

from typing import Any, Dict, List

from .base import PackResult, GauntletPack


class ScalabilityPack(GauntletPack):
    name = "scalability"

    def inject(self, state: Dict[str, Any], step_ctx: Dict[str, Any]) -> Dict[str, Any]:
        load_profile = step_ctx.get("load_profile", {})
        step_ctx["concurrency"] = int(load_profile.get("concurrency", 1))
        step_ctx["queue_depth"] = int(load_profile.get("queue_depth", 0))
        return step_ctx

    def verify(self, action: Any, obs: Any, gt: Dict[str, Any]) -> PackResult:
        latency = float(getattr(obs, "last_step_latency_ms", 0.0) or 0.0)
        concurrency = int(gt.get("concurrency", 1))
        queue_depth = int(gt.get("queue_depth", 0))
        budget_remaining = float(getattr(obs, "budget_remaining", 1.0) or 1.0)
        sla_limit = float(getattr(obs, "sla_limit_ms", 5000.0) or 5000.0)
        throughput_stability = 1.0 if latency <= sla_limit else 0.0
        sla_under_load = 1.0 if latency <= sla_limit * 1.2 else 0.0
        cost_eff = max(0.0, min(1.0, budget_remaining))
        violations: List[str] = []
        if queue_depth > 100 and action.action_type == "call_tool":
            violations.append("backpressure_ignored")
        if concurrency >= 50 and latency > sla_limit:
            violations.append("sla_breach_under_load")
        return PackResult(
            subscores={
                "sla_under_load_score": sla_under_load,
                "cost_efficiency_score": cost_eff,
                "throughput_stability_score": throughput_stability,
            },
            violations=violations,
            evidence={
                "latency_ms": latency,
                "concurrency": concurrency,
                "queue_depth": queue_depth,
            },
        )

    def metrics(self, episode_trace: List[Dict[str, Any]]) -> Dict[str, float]:
        if not episode_trace:
            return {"throughput": 0.0, "sla_compliance_under_load": 0.0, "cost_reward_efficiency_at_scale": 0.0}
        latencies = [float(r.get("latency_ms", 0.0)) for r in episode_trace]
        total = max(1, len(latencies))
        under_sla = sum(1 for l in latencies if l <= 5000.0)
        avg_reward = sum(float(r.get("reward", 0.0)) for r in episode_trace) / total
        return {
            "throughput": total / max(1.0, sum(latencies) / 1000.0),
            "sla_compliance_under_load": under_sla / total,
            "cost_reward_efficiency_at_scale": avg_reward,
        }
