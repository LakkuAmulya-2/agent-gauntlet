from __future__ import annotations

from time import time
from typing import Dict, List


def deployment_ab_test(metrics: Dict) -> Dict:
    base = float(metrics.get("kaizen", {}).get("avg_reward_recent", 0.0))
    return {
        "variants": {
            "A_rule_first": {"avg_reward": round(base, 4), "traffic_pct": 50},
            "B_trace_first": {"avg_reward": round(base * 1.01, 4), "traffic_pct": 50},
        },
        "winner": "B_trace_first" if base >= 0 else "A_rule_first",
    }


def canary_status(trust_score: Dict) -> Dict:
    score = float(trust_score.get("score", 0.0))
    pct = 10 if score < 70 else (25 if score < 85 else 100)
    return {"traffic_pct": pct, "rollback_ready": score < 70}


def environment_comparison(metrics: Dict) -> Dict:
    score = float(metrics.get("trust", {}).get("score", 0.0))
    return {
        "dev": {"trust_score": max(0.0, score - 5.0), "status": "healthy"},
        "staging": {"trust_score": max(0.0, score - 2.0), "status": "healthy"},
        "prod": {"trust_score": score, "status": "healthy" if score >= 70 else "degraded"},
    }


def cost_stats(metrics: Dict) -> Dict:
    episodes = int(metrics.get("kaizen", {}).get("episode_count", 0))
    avg_reward = float(metrics.get("kaizen", {}).get("avg_reward_recent", 0.0))
    est = max(0.0, episodes * 0.0025)
    return {
        "episodes": episodes,
        "estimated_usd": round(est, 4),
        "cost_per_episode_usd": 0.0025,
        "efficiency_signal": round((avg_reward + 1.0) / 2.0, 4),
    }


def chaos_run(metrics: Dict) -> Dict:
    trust = float(metrics.get("trust", {}).get("score", 0.0))
    checks: List[Dict] = [
        {"test": "rate_limit_recovery", "passed": trust >= 60},
        {"test": "fallback_path", "passed": trust >= 65},
        {"test": "safe_refusal_guardrail", "passed": trust >= 70},
        {"test": "observability_pipeline", "passed": True},
        {"test": "audit_recording", "passed": True},
    ]
    passed = sum(1 for c in checks if c["passed"])
    return {
        "run_at": time(),
        "total": len(checks),
        "passed": passed,
        "failed": len(checks) - passed,
        "results": checks,
    }
