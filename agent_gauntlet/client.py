# Copyright (c) 2026 Agent Gauntlet Contributors
# BSD-style license

"""
Client for Agent Gauntlet environment.

Handles HTTP/WebSocket communication with the environment server.
Supports both async and sync usage patterns.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient

# StepResult is the return type of EnvClient.step() — defined inline for compatibility
from typing import Generic, TypeVar, NamedTuple

_O = TypeVar("_O")

class StepResult(NamedTuple):
    """Minimal StepResult compatible with openenv.core EnvClient."""
    observation: Any
    reward: float
    done: bool

from .models import AgentAction, EpisodeState, TaskObservation, ToolResult


class AgentGauntletEnv(EnvClient[AgentAction, TaskObservation, EpisodeState]):
    """
    Client for the Agent Gauntlet RL environment.

    Trains LLMs to handle real production failure conditions:
    - API failures (500, 429, 401, timeouts)
    - Cascading failures across multi-step workflows
    - Context pressure over long-horizon tasks
    - Adversarial inputs and malformed data
    - Resource/budget overruns

    Usage (async):
        async with AgentGauntletEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(AgentAction(
                action_type="call_tool",
                tool_name="fetch_records",
                tool_args={"endpoint": "/api/users"},
                reasoning="Starting data pipeline by fetching user records"
            ))
            print(f"Reward: {result.reward}")

    Usage (sync):
        with AgentGauntletEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            result = env.step(AgentAction(...))
    """

    def _step_payload(self, action: AgentAction) -> Dict[str, Any]:
        """Convert AgentAction to JSON payload for HTTP request."""
        return {
            "action_type": action.action_type,
            "tool_name": action.tool_name,
            "tool_args": action.tool_args or {},
            "reasoning": action.reasoning or "",
            "failure_detected": action.failure_detected,
            "recovery_strategy": action.recovery_strategy,
            "escalation_reason": action.escalation_reason,
            "task_result": action.task_result,
            "target_agent_id": action.target_agent_id,
            "message_content": action.message_content,
            "subtask_description": action.subtask_description,
            "harder_variant_description": action.harder_variant_description,
            "state_summary": action.state_summary,
            "drift_detected": action.drift_detected,
            "contradiction_resolution": action.contradiction_resolution,
            # Security
            "injection_refused": action.injection_refused,
            "injection_description": action.injection_description,
            # Compliance
            "compliance_check_result": action.compliance_check_result,
            "compliance_policy": action.compliance_policy,
            "compliance_alternative": action.compliance_alternative,
            "decision_documented": action.decision_documented,
            # Observability / Theme #4
            "diagnostic_trace": action.diagnostic_trace,
            # Theory of Mind / Theme #1
            "stakeholder_belief_update": action.stakeholder_belief_update,
            "transparency_decision": action.transparency_decision,
            # Long-horizon / Theme #2
            "checkpoint_data": action.checkpoint_data,
            "checkpoint_id": action.checkpoint_id,
            # Reliability / calibration
            "idempotency_key": action.idempotency_key,
            "risk_acknowledged": action.risk_acknowledged,
            "confidence_score": action.confidence_score,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TaskObservation]:
        """Parse HTTP response into typed StepResult."""
        obs_data = payload.get("observation", {})

        # Parse nested ToolResult if present
        tool_result_data = obs_data.get("last_tool_result")
        tool_result = None
        if tool_result_data:
            tool_result = ToolResult(**tool_result_data)

        obs = TaskObservation(
            task_description=obs_data.get("task_description", ""),
            task_domain=obs_data.get("task_domain", "data_pipeline"),
            available_tools=obs_data.get("available_tools", []),
            task_goal=obs_data.get("task_goal", ""),
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 50),
            last_tool_result=tool_result,
            context_used_pct=obs_data.get("context_used_pct", 0.0),
            budget_remaining=obs_data.get("budget_remaining", 1.0),
            api_calls_made=obs_data.get("api_calls_made", 0),
            api_calls_budget=obs_data.get("api_calls_budget", 20),
            recent_errors=obs_data.get("recent_errors", []),
            consecutive_failures=obs_data.get("consecutive_failures", 0),
            other_agents_status=obs_data.get("other_agents_status", {}),
            incoming_messages=obs_data.get("incoming_messages", []),
            delegated_subtasks=obs_data.get("delegated_subtasks", []),
            completed_checkpoints=obs_data.get("completed_checkpoints", []),
            pending_objectives=obs_data.get("pending_objectives", []),
            state_summary=obs_data.get("state_summary"),
            episode_id=obs_data.get("episode_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            is_done=obs_data.get("is_done", False),
            termination_reason=obs_data.get("termination_reason"),
            metadata=obs_data.get("metadata", {}),
            token_cost_used_usd=obs_data.get("token_cost_used_usd", 0.0),
            token_budget_usd=obs_data.get("token_budget_usd", 1.0),
            cost_overrun_warning=obs_data.get("cost_overrun_warning", False),
            # Security
            security_alert=obs_data.get("security_alert"),
            injection_attempt_count=obs_data.get("injection_attempt_count", 0),
            # Compliance
            active_policies=obs_data.get("active_policies", []),
            compliance_warnings=obs_data.get("compliance_warnings", []),
            # SLA
            sla_limit_ms=obs_data.get("sla_limit_ms", 5000.0),
            last_step_latency_ms=obs_data.get("last_step_latency_ms", 0.0),
            sla_breaches=obs_data.get("sla_breaches", 0),
            # Observability
            episode_traces=obs_data.get("episode_traces", []),
            # Theory of Mind
            stakeholder_belief=obs_data.get("stakeholder_belief"),
            stakeholder_expectation=obs_data.get("stakeholder_expectation"),
            # Long-horizon
            context_checkpoint_available=obs_data.get("context_checkpoint_available", False),
            last_checkpoint_step=obs_data.get("last_checkpoint_step", 0),
            total_steps_including_resumed=obs_data.get("total_steps_including_resumed", 0),
            # Memory — past lessons from Forge kernel
            past_lessons=obs_data.get("past_lessons", []),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=obs_data.get("is_done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EpisodeState:
        """Parse HTTP response into typed EpisodeState."""
        return EpisodeState(
            task_id=payload.get("task_id", ""),
            task_domain=payload.get("task_domain", "data_pipeline"),
            difficulty=payload.get("difficulty", "easy"),
            task_completed=payload.get("task_completed", False),
            task_result_correct=payload.get("task_result_correct", False),
            injected_failures=payload.get("injected_failures", []),
            failures_detected_correctly=payload.get("failures_detected_correctly", 0),
            failures_missed=payload.get("failures_missed", 0),
            false_positive_detections=payload.get("false_positive_detections", 0),
            recovery_attempts=payload.get("recovery_attempts", 0),
            successful_recoveries=payload.get("successful_recoveries", 0),
            escalations=payload.get("escalations", 0),
            correct_escalations=payload.get("correct_escalations", 0),
            unnecessary_escalations=payload.get("unnecessary_escalations", 0),
            total_api_calls=payload.get("total_api_calls", 0),
            api_calls_budget=payload.get("api_calls_budget", 20),
            budget_used_pct=payload.get("budget_used_pct", 0.0),
            budget_remaining=payload.get("budget_remaining", 1.0),
            context_pressure_events=payload.get("context_pressure_events", 0),
            cascade_triggered=payload.get("cascade_triggered", False),
            cascade_caught=payload.get("cascade_caught", False),
            messages_sent=payload.get("messages_sent", 0),
            subtasks_delegated=payload.get("subtasks_delegated", 0),
            subtasks_completed=payload.get("subtasks_completed", 0),
            checkpoints_reached=payload.get("checkpoints_reached", []),
            objectives_completed=payload.get("objectives_completed", 0),
            total_objectives=payload.get("total_objectives", 0),
            harder_variants_generated=payload.get("harder_variants_generated", 0),
            step_count=payload.get("step_count", 0),
            max_steps=payload.get("max_steps", 50),
            consecutive_failures=payload.get("consecutive_failures", 0),
            step_rewards=payload.get("step_rewards", []),
            semantic_drifts_injected=payload.get("semantic_drifts_injected", 0),
            semantic_drifts_detected=payload.get("semantic_drifts_detected", 0),
            contradictions_detected=payload.get("contradictions_detected", 0),
            token_budget_usd=payload.get("token_budget_usd", 1.0),
            token_cost_used_usd=payload.get("token_cost_used_usd", 0.0),
            cost_overrun=payload.get("cost_overrun", False),
            # Security
            security_breaches_injected=payload.get("security_breaches_injected", 0),
            security_breaches_detected=payload.get("security_breaches_detected", 0),
            injections_refused=payload.get("injections_refused", 0),
            injections_followed=payload.get("injections_followed", 0),
            # Compliance
            compliance_violations_injected=payload.get("compliance_violations_injected", 0),
            compliance_violations_detected=payload.get("compliance_violations_detected", 0),
            compliant_alternatives_found=payload.get("compliant_alternatives_found", 0),
            decisions_documented=payload.get("decisions_documented", 0),
            # SLA
            sla_limit_ms=payload.get("sla_limit_ms", 5000.0),
            sla_breaches=payload.get("sla_breaches", 0),
            total_step_latency_ms=payload.get("total_step_latency_ms", 0.0),
            # Observability
            diagnostic_traces=payload.get("diagnostic_traces", []),
            trace_quality_scores=payload.get("trace_quality_scores", []),
            # Theory of Mind
            stakeholder_belief_state=payload.get("stakeholder_belief_state", "unknown"),
            tom_correct_decisions=payload.get("tom_correct_decisions", 0),
            tom_incorrect_decisions=payload.get("tom_incorrect_decisions", 0),
            # Long-horizon
            checkpoints_saved=payload.get("checkpoints_saved", 0),
            checkpoints_resumed=payload.get("checkpoints_resumed", 0),
            context_resets=payload.get("context_resets", 0),
            state_recall_scores=payload.get("state_recall_scores", []),
        )
