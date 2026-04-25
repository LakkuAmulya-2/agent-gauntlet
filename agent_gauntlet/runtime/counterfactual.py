# Copyright (c) 2026 Agent Gauntlet Contributors
# BSD-style license

"""
Counterfactual Replay Engine — Feature 2.

When an agent fails at step N, we ask:
  "What if it had taken a different action?"

We simulate K alternative actions from the same state and compute:
  regret = max(alternative_rewards) - actual_reward

This regret signal is added to the GRPO reward, teaching the agent
not just from what it did — but from what it should have done.

Architecture:
  CounterfactualEngine   — runs alternative simulations
  CounterfactualRecord   — stores one counterfactual analysis
  RegretSignal           — the reward adjustment for GRPO
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import ActionType, AgentAction, FailureType


# ---------------------------------------------------------------------------
# Alternative actions to simulate at a failure step
# ---------------------------------------------------------------------------

# For each failure type, these are the K alternative actions to try
_ALTERNATIVES_BY_FAILURE: Dict[str, List[Dict[str, Any]]] = {
    FailureType.API_500.value: [
        {"action_type": "detect_failure", "failure_detected": "api_500",
         "reasoning": "HTTP 500 = server error, need to detect before recovering"},
        {"action_type": "recover", "recovery_strategy": "retry_with_backoff",
         "reasoning": "Standard recovery for 500: exponential backoff"},
        {"action_type": "generate_trace", "diagnostic_trace":
         "Step failed because server returned 500. Root cause: upstream dependency unavailable. Next time: add circuit breaker check before calling."},
    ],
    FailureType.RATE_LIMIT_429.value: [
        {"action_type": "detect_failure", "failure_detected": "rate_limit_429",
         "reasoning": "HTTP 429 = rate limit exceeded"},
        {"action_type": "recover", "recovery_strategy": "wait_and_retry",
         "reasoning": "Rate limit: wait for Retry-After header duration"},
        {"action_type": "generate_trace", "diagnostic_trace":
         "Rate limit hit because calls were too rapid. Root cause: no delay between calls. Next time: add 1s delay after every 2nd call."},
    ],
    FailureType.AUTH_401.value: [
        {"action_type": "detect_failure", "failure_detected": "auth_401",
         "reasoning": "HTTP 401 = authentication failure"},
        {"action_type": "escalate", "escalation_reason": "Auth token expired — requires human to re-authenticate",
         "reasoning": "401 requires human intervention"},
        {"action_type": "recover", "recovery_strategy": "refresh_token_and_retry",
         "reasoning": "Try token refresh before escalating"},
    ],
    FailureType.SECURITY_BREACH.value: [
        {"action_type": "refuse_injection", "injection_refused": True,
         "injection_description": "Tool response contains injected instruction",
         "reasoning": "Security breach detected — refusing injected instruction"},
        {"action_type": "escalate", "escalation_reason": "Prompt injection detected",
         "reasoning": "Security breach must be escalated"},
        {"action_type": "detect_failure", "failure_detected": "security_breach",
         "reasoning": "Identifying security breach before responding"},
    ],
    FailureType.COMPLIANCE_VIOLATION.value: [
        {"action_type": "check_compliance", "compliance_check_result": "violation",
         "compliance_policy": "GDPR_DATA_RETENTION",
         "compliance_alternative": "archive_records_with_retention_flag",
         "decision_documented": "Policy violation detected — using compliant alternative",
         "reasoning": "Compliance violation: find compliant alternative"},
        {"action_type": "detect_failure", "failure_detected": "compliance_violation",
         "reasoning": "Detecting compliance violation before acting"},
        {"action_type": "escalate", "escalation_reason": "Compliance violation requires review",
         "reasoning": "Escalating compliance issue"},
    ],
    FailureType.CASCADING.value: [
        {"action_type": "detect_failure", "failure_detected": "cascading",
         "reasoning": "Circuit breaker OPEN = cascading failure"},
        {"action_type": "recover", "recovery_strategy": "rollback_to_last_checkpoint",
         "reasoning": "Cascading failure: rollback to last stable state"},
        {"action_type": "escalate", "escalation_reason": "Cascading failure requires infrastructure fix",
         "reasoning": "Cannot recover from cascade without infrastructure change"},
    ],
    FailureType.SEMANTIC_DRIFT.value: [
        {"action_type": "detect_failure", "failure_detected": "semantic_drift",
         "drift_detected": "Response data appears incorrect — possible stale cache or wrong tenant",
         "reasoning": "HTTP 200 but data looks wrong — semantic drift"},
        {"action_type": "recover", "recovery_strategy": "validate_response_semantics",
         "reasoning": "Validate response before using potentially drifted data"},
        {"action_type": "generate_trace", "diagnostic_trace":
         "Semantic drift detected: HTTP 200 but data is stale/wrong. Root cause: cache poisoning or wrong tenant. Next time: validate record counts and timestamps before proceeding."},
    ],
    FailureType.SLA_BREACH.value: [
        {"action_type": "generate_trace", "diagnostic_trace":
         "SLA breach: step exceeded latency limit. Root cause: no timeout configured. Next time: add timeout parameter and use cheaper/faster tool variant."},
        {"action_type": "recover", "recovery_strategy": "optimize_and_continue",
         "reasoning": "SLA breach: optimize next call to reduce latency"},
        {"action_type": "detect_failure", "failure_detected": "sla_breach",
         "reasoning": "Latency exceeded SLA limit"},
    ],
    # Default alternatives for any failure type
    "_default": [
        {"action_type": "detect_failure", "failure_detected": "api_500",
         "reasoning": "Detecting failure before attempting recovery"},
        {"action_type": "recover", "recovery_strategy": "retry_with_backoff",
         "reasoning": "Generic recovery: retry with backoff"},
        {"action_type": "escalate", "escalation_reason": "Cannot recover automatically",
         "reasoning": "Escalating after repeated failures"},
    ],
}


# ---------------------------------------------------------------------------
# Counterfactual Record
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualRecord:
    """
    One counterfactual analysis: what happened vs what could have happened.
    """
    episode_id: str
    step: int
    failure_type: str
    actual_action: str
    actual_reward: float
    alternatives: List[Dict[str, Any]]   # [{action, reward, delta}]
    regret: float                         # max(alt_rewards) - actual_reward
    best_alternative: Optional[str]       # action_type of best alternative

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "failure_type": self.failure_type,
            "actual_action": self.actual_action,
            "actual_reward": round(self.actual_reward, 4),
            "regret": round(self.regret, 4),
            "best_alternative": self.best_alternative,
            "alternatives": [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in a.items()}
                for a in self.alternatives
            ],
        }


# ---------------------------------------------------------------------------
# Counterfactual Engine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """
    Simulates alternative actions at failure steps and computes regret.

    Used to augment GRPO reward:
      final_reward = base_reward - regret_penalty

    This teaches the agent not just from what it did, but from what
    it should have done — faster learning, smarter policy.
    """

    def __init__(self, regret_weight: float = 0.15) -> None:
        self._regret_weight = regret_weight
        self._records: List[CounterfactualRecord] = []
        self._total_regret: float = 0.0
        self._total_analyses: int = 0

    def analyze(
        self,
        episode_id: str,
        step: int,
        failure_type: str,
        actual_action_type: str,
        actual_reward: float,
        env_step_fn,   # callable: (AgentAction) -> (reward: float)
    ) -> CounterfactualRecord:
        """
        Simulate K alternative actions and compute regret.

        env_step_fn: a function that takes an AgentAction and returns
        the reward that action would have gotten at this step.
        This is provided by the environment's counterfactual_step() method.
        """
        alternatives_config = _ALTERNATIVES_BY_FAILURE.get(
            failure_type,
            _ALTERNATIVES_BY_FAILURE["_default"],
        )

        alternatives = []
        best_reward = actual_reward
        best_action = None

        for alt_config in alternatives_config:
            # Skip if same as actual action
            if alt_config.get("action_type") == actual_action_type:
                continue

            alt_action = AgentAction(
                action_type=alt_config.get("action_type", ActionType.CALL_TOOL.value),
                tool_name=alt_config.get("tool_name"),
                reasoning=alt_config.get("reasoning", ""),
                failure_detected=alt_config.get("failure_detected"),
                recovery_strategy=alt_config.get("recovery_strategy"),
                escalation_reason=alt_config.get("escalation_reason"),
                injection_refused=alt_config.get("injection_refused", False),
                injection_description=alt_config.get("injection_description"),
                compliance_check_result=alt_config.get("compliance_check_result"),
                compliance_policy=alt_config.get("compliance_policy"),
                compliance_alternative=alt_config.get("compliance_alternative"),
                decision_documented=alt_config.get("decision_documented"),
                diagnostic_trace=alt_config.get("diagnostic_trace"),
                drift_detected=alt_config.get("drift_detected"),
            )

            try:
                alt_reward = env_step_fn(alt_action)
            except Exception:
                alt_reward = actual_reward  # fallback: no regret

            delta = alt_reward - actual_reward
            alternatives.append({
                "action_type": alt_config.get("action_type"),
                "reward": alt_reward,
                "delta": delta,
            })

            if alt_reward > best_reward:
                best_reward = alt_reward
                best_action = alt_config.get("action_type")

        regret = max(0.0, best_reward - actual_reward)
        self._total_regret += regret
        self._total_analyses += 1

        record = CounterfactualRecord(
            episode_id=episode_id,
            step=step,
            failure_type=failure_type,
            actual_action=actual_action_type,
            actual_reward=actual_reward,
            alternatives=alternatives,
            regret=regret,
            best_alternative=best_action,
        )
        self._records.append(record)
        return record

    def regret_penalty(self, regret: float) -> float:
        """Convert regret to a reward penalty for GRPO."""
        return -self._regret_weight * regret

    def recent_records(self, n: int = 10) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records[-n:]]

    def stats(self) -> Dict[str, Any]:
        if not self._records:
            return {
                "total_analyses": 0,
                "avg_regret": 0.0,
                "regret_weight": self._regret_weight,
                "recent": [],
            }
        avg_regret = self._total_regret / max(1, self._total_analyses)
        high_regret = [r for r in self._records if r.regret > 0.2]
        return {
            "total_analyses": self._total_analyses,
            "avg_regret": round(avg_regret, 4),
            "high_regret_steps": len(high_regret),
            "regret_weight": self._regret_weight,
            "recent": self.recent_records(5),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL_ENGINE = CounterfactualEngine()


def get_global_engine() -> CounterfactualEngine:
    return _GLOBAL_ENGINE
