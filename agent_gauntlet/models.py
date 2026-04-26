# Copyright (c) 2026 Agent Gauntlet Contributors
# BSD-style license

"""
Type-safe models for Agent Gauntlet environment.

All actions, observations, and state are strongly typed dataclasses.
No hardcoded dummy data — all scenarios are procedurally generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import TypeAdapter

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All valid action types an agent can take."""
    CALL_TOOL = "call_tool"
    DETECT_FAILURE = "detect_failure"
    RECOVER = "recover"
    ESCALATE = "escalate"
    SUMMARIZE_STATE = "summarize_state"
    COMPLETE_TASK = "complete_task"
    # Multi-agent / Theory of Mind
    SEND_MESSAGE = "send_message"
    DELEGATE_SUBTASK = "delegate_subtask"
    REQUEST_STATUS = "request_status"
    UPDATE_STAKEHOLDER_BELIEF = "update_stakeholder_belief"  # Theme #1 ToM
    INFORM_STAKEHOLDER = "inform_stakeholder"                # Theme #1 ToM — transparency
    # Self-improvement / Observability
    GENERATE_HARDER_VARIANT = "generate_harder_variant"
    GENERATE_TRACE = "generate_trace"                        # Theme #4 — diagnostic trace
    # Semantic drift / contradiction
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    # Security
    REFUSE_INJECTION = "refuse_injection"                    # Refuse injected instruction
    # Compliance
    CHECK_COMPLIANCE = "check_compliance"                    # Verify action against policy
    DOCUMENT_DECISION = "document_decision"                  # Document compliance decision
    # Long-horizon / Scalability
    CHECKPOINT_STATE = "checkpoint_state"                    # Theme #2 — save checkpoint
    RESUME_FROM_CHECKPOINT = "resume_from_checkpoint"        # Theme #2 — resume after context reset


class FailureType(str, Enum):
    """Production failure modes injected by the environment."""
    API_500 = "api_500"
    RATE_LIMIT_429 = "rate_limit_429"
    AUTH_401 = "auth_401"
    MALFORMED_RESPONSE = "malformed_response"
    TIMEOUT = "timeout"
    CASCADING = "cascading"
    ADVERSARIAL_INPUT = "adversarial_input"
    CONTEXT_PRESSURE = "context_pressure"
    SEMANTIC_DRIFT = "semantic_drift"          # HTTP 200 but semantically wrong data
    COST_OVERRUN = "cost_overrun"              # Token budget exceeded
    SECURITY_BREACH = "security_breach"        # Prompt injection / data poisoning
    COMPLIANCE_VIOLATION = "compliance_violation"  # Policy violation detected
    SLA_BREACH = "sla_breach"                  # Step exceeded SLA time limit
    NONE = "none"


class TaskDomain(str, Enum):
    """Enterprise task domains the agent must complete."""
    DATA_PIPELINE = "data_pipeline"
    API_WORKFLOW = "api_workflow"
    REPORT_GENERATION = "report_generation"
    SYSTEM_CONFIG = "system_config"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    CODE_REVIEW = "code_review"
    INCIDENT_RESPONSE = "incident_response"
    PERSONAL_ASSISTANT = "personal_assistant"
    LARGE_SCALE_MIGRATION = "large_scale_migration"  # Theme #2 — 200+ steps, context compression


class DifficultyLevel(str, Enum):
    """Curriculum difficulty levels."""
    EASY = "easy"       # 1-2 failures, short horizon (10 steps)
    MEDIUM = "medium"   # 3-4 failures, medium horizon (25 steps)
    HARD = "hard"       # 5+ failures, long horizon (50 steps)
    EXPERT = "expert"   # Cascading + adversarial, 50+ steps


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """
    Action taken by the agent in the Agent Gauntlet.

    The agent can call tools, detect failures, recover, escalate,
    compress context, coordinate with other agents, or declare task completion.
    """
    action_type: str = ActionType.CALL_TOOL.value
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    failure_detected: Optional[str] = None   # FailureType if detecting
    recovery_strategy: Optional[str] = None  # How agent plans to recover
    escalation_reason: Optional[str] = None  # Why escalating
    task_result: Optional[str] = None        # Final answer if completing
    # Multi-agent fields
    target_agent_id: Optional[str] = None    # Which agent to communicate with
    message_content: Optional[str] = None    # Message to send
    subtask_description: Optional[str] = None  # Subtask to delegate
    # Self-improvement fields
    harder_variant_description: Optional[str] = None  # Proposed harder challenge
    # Long-horizon state management
    state_summary: Optional[str] = None      # Compressed state summary
    # Semantic drift / contradiction fields
    drift_detected: Optional[str] = None
    contradiction_resolution: Optional[str] = None
    # Security fields
    injection_refused: bool = False              # Agent refused injected instruction
    injection_description: Optional[str] = None  # What injection was detected
    # Compliance fields
    compliance_check_result: Optional[str] = None  # "compliant" | "violation" | "alternative_found"
    compliance_policy: Optional[str] = None         # Which policy was checked
    compliance_alternative: Optional[str] = None    # Compliant alternative action
    decision_documented: Optional[str] = None       # Documentation of compliance decision
    # Observability / Theme #4
    diagnostic_trace: Optional[str] = None          # Agent's self-generated trace
    # Theory of Mind / Theme #1
    stakeholder_belief_update: Optional[str] = None  # What agent believes stakeholder thinks
    transparency_decision: Optional[str] = None       # "inform" | "silent_fix" | "escalate"
    # Long-horizon / Theme #2
    checkpoint_data: Optional[str] = None            # Serialized checkpoint state
    checkpoint_id: Optional[str] = None              # ID of checkpoint to resume from
    # Reliability / Guardrail
    idempotency_key: Optional[str] = None
    risk_acknowledged: Optional[str] = None
    confidence_score: Optional[float] = None   # 0..1 self-reported confidence

    def __post_init__(self) -> None:
        # Normalize enum inputs for callers that pass ActionType directly.
        if isinstance(self.action_type, ActionType):
            self.action_type = self.action_type.value

        allowed = {x.value for x in ActionType}
        if self.action_type not in allowed:
            raise ValueError(
                f"Invalid action_type='{self.action_type}'. Must be one of: {sorted(allowed)}"
            )

        if self.confidence_score is not None:
            self.confidence_score = float(self.confidence_score)
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError("confidence_score must be between 0 and 1.")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Compatibility shim for OpenEnv /schema endpoint."""
        return TypeAdapter(cls).json_schema()

    @classmethod
    def model_validate(cls, payload: Any) -> "AgentAction":
        """Pydantic-compatible constructor used by OpenEnv internals."""
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            return cls(**payload)
        raise TypeError(f"Unsupported payload type for AgentAction: {type(payload)!r}")

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Pydantic-compatible serializer used by OpenEnv internals."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of a tool/API call — may contain injected failures."""
    tool_name: str
    success: bool
    status_code: int
    response: Optional[Dict[str, Any]]
    error_message: Optional[str]
    latency_ms: float
    failure_type: str = FailureType.NONE.value


@dataclass
class TaskObservation:
    """
    What the agent sees at each step.

    Includes task context, last tool result, current resource usage,
    multi-agent messages, and hints about the environment state.
    """
    # Task context
    task_description: str = ""
    task_domain: str = TaskDomain.DATA_PIPELINE.value
    available_tools: List[str] = field(default_factory=list)
    task_goal: str = ""

    # Current step info
    current_step: int = 0
    max_steps: int = 50
    last_tool_result: Optional[ToolResult] = None

    # Resource tracking
    context_used_pct: float = 0.0    # 0.0 to 1.0
    budget_remaining: float = 1.0    # 0.0 to 1.0 (normalized)
    api_calls_made: int = 0
    api_calls_budget: int = 20

    # Failure signals (partial — agent must infer full picture)
    recent_errors: List[str] = field(default_factory=list)
    consecutive_failures: int = 0

    # Multi-agent context
    other_agents_status: Dict[str, str] = field(default_factory=dict)
    incoming_messages: List[Dict[str, str]] = field(default_factory=list)  # messages from other agents
    delegated_subtasks: List[Dict[str, Any]] = field(default_factory=list)  # subtask results

    # Long-horizon state tracking
    completed_checkpoints: List[str] = field(default_factory=list)  # what's been done
    pending_objectives: List[str] = field(default_factory=list)     # what remains
    state_summary: Optional[str] = None  # compressed state for context management

    # Episode metadata
    episode_id: str = ""
    difficulty: str = DifficultyLevel.EASY.value

    # Terminal signal
    is_done: bool = False
    termination_reason: Optional[str] = None

    # Reward metadata (for TRL)
    metadata: Dict[str, Any] = field(default_factory=dict)
    verifier_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Token cost tracking
    token_cost_used_usd: float = 0.0
    token_budget_usd: float = 1.0
    cost_overrun_warning: bool = False

    # Security signals
    security_alert: Optional[str] = None        # Injected payload description (partial)
    injection_attempt_count: int = 0

    # Compliance signals
    active_policies: List[str] = field(default_factory=list)  # Policies in effect
    compliance_warnings: List[str] = field(default_factory=list)

    # SLA tracking
    sla_limit_ms: float = 5000.0             # SLA per step in ms
    last_step_latency_ms: float = 0.0
    sla_breaches: int = 0

    # Observability / Theme #4
    episode_traces: List[str] = field(default_factory=list)  # Traces generated this episode

    # Theory of Mind / Theme #1
    stakeholder_belief: Optional[str] = None   # What stakeholder currently believes
    stakeholder_expectation: Optional[str] = None  # What stakeholder expects next

    # Long-horizon / Theme #2
    context_checkpoint_available: bool = False
    last_checkpoint_step: int = 0
    total_steps_including_resumed: int = 0

    # Memory — past lessons from Forge kernel, visible to agent at episode start
    past_lessons: List[str] = field(default_factory=list)

    def assert_valid(self) -> None:
        """Optional strict checker for debugging and tests."""
        if self.current_step < 0:
            raise ValueError("current_step must be >= 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if not (0.0 <= self.context_used_pct <= 1.0):
            raise ValueError("context_used_pct must be between 0 and 1")
        if not (0.0 <= self.budget_remaining <= 1.0):
            raise ValueError("budget_remaining must be between 0 and 1")
        if self.task_domain not in {x.value for x in TaskDomain}:
            raise ValueError(f"Unknown task_domain: {self.task_domain}")
        if self.difficulty not in {x.value for x in DifficultyLevel}:
            raise ValueError(f"Unknown difficulty: {self.difficulty}")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Compatibility shim for OpenEnv /schema endpoint."""
        return TypeAdapter(cls).json_schema()

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Pydantic-compatible serializer used by OpenEnv internals."""
        return asdict(self)

    @property
    def observation(self) -> "TaskObservation":
        """StepResult compatibility: OpenEnv clients may expect `.observation`."""
        return self

    @property
    def reward(self) -> float:
        """StepResult compatibility: expose per-step reward when present."""
        return float(getattr(self, "_reward", 0.0))

    @property
    def done(self) -> bool:
        """StepResult compatibility: OpenEnv clients may expect `.done`."""
        return bool(self.is_done)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """
    Full internal episode state (server-side).

    Tracks ground truth for reward computation.
    """
    # Task tracking
    task_id: str = ""
    task_domain: str = TaskDomain.DATA_PIPELINE.value
    difficulty: str = DifficultyLevel.EASY.value
    task_completed: bool = False
    task_result_correct: bool = False

    # Failure tracking (ground truth)
    injected_failures: List[Dict[str, Any]] = field(default_factory=list)
    failures_detected_correctly: int = 0
    failures_missed: int = 0
    false_positive_detections: int = 0

    # Recovery tracking
    recovery_attempts: int = 0
    successful_recoveries: int = 0

    # Escalation tracking
    escalations: int = 0
    correct_escalations: int = 0
    unnecessary_escalations: int = 0

    # Resource tracking
    total_api_calls: int = 0
    api_calls_budget: int = 20
    budget_used_pct: float = 0.0
    budget_remaining: float = 1.0   # FIX: kept in sync with budget_used_pct
    context_pressure_events: int = 0

    # Cascade tracking
    cascade_triggered: bool = False
    cascade_caught: bool = False

    # Multi-agent tracking
    messages_sent: int = 0
    subtasks_delegated: int = 0
    subtasks_completed: int = 0
    agent_messages: List[Dict[str, str]] = field(default_factory=list)
    coordination_conflicts_detected: int = 0
    coordination_conflicts_resolved: int = 0

    # Long-horizon tracking
    checkpoints_reached: List[str] = field(default_factory=list)
    state_summaries: List[str] = field(default_factory=list)
    objectives_completed: int = 0
    total_objectives: int = 0

    # Self-improvement tracking
    harder_variants_generated: int = 0
    difficulty_escalations: int = 0

    # Step tracking
    step_count: int = 0
    max_steps: int = 50
    consecutive_failures: int = 0

    # Reward hacking prevention — use non-private name to avoid dataclass issues (Fix #19)
    last_action_key: Optional[str] = None
    repeated_action_count: int = 0

    # Reward components (tracked per step for process supervision)
    step_rewards: List[float] = field(default_factory=list)

    # Semantic drift / contradiction tracking
    semantic_drifts_injected: int = 0
    semantic_drifts_detected: int = 0
    contradictions_detected: int = 0
    token_budget_usd: float = 1.0
    token_cost_used_usd: float = 0.0
    cost_overrun: bool = False

    # Security tracking (Theme #3.1 + #5)
    security_breaches_injected: int = 0
    security_breaches_detected: int = 0
    injections_refused: int = 0
    injections_followed: int = 0          # Agent followed injected instruction = bad

    # Compliance tracking (Theme #3.1)
    compliance_violations_injected: int = 0
    compliance_violations_detected: int = 0
    compliant_alternatives_found: int = 0
    decisions_documented: int = 0

    # SLA / Reliability tracking (Theme #3.1 + #5)
    sla_limit_ms: float = 5000.0
    sla_breaches: int = 0
    total_step_latency_ms: float = 0.0

    # Observability / Theme #4 — self-improvement traces
    diagnostic_traces: List[str] = field(default_factory=list)
    trace_quality_scores: List[float] = field(default_factory=list)

    # Theory of Mind / Theme #1
    stakeholder_belief_state: str = "unknown"   # What stakeholder believes
    tom_correct_decisions: int = 0              # Correct transparency decisions
    tom_incorrect_decisions: int = 0

    # Long-horizon / Theme #2 — context compression
    checkpoints_saved: int = 0
    checkpoints_resumed: int = 0
    context_resets: int = 0
    state_recall_scores: List[float] = field(default_factory=list)
    last_checkpoint_step: int = 0   # Fix #4: track which step the last checkpoint was saved
    # Deterministic replay + auditability
    scenario_profile: Dict[str, Any] = field(default_factory=dict)
    perturbation_profile: Dict[str, Any] = field(default_factory=dict)
    episode_trace: List[Dict[str, Any]] = field(default_factory=list)
    policy_decisions: List[Dict[str, Any]] = field(default_factory=list)
    step_latencies_ms: List[float] = field(default_factory=list)
    tool_costs_usd: List[float] = field(default_factory=list)
    idempotency_seen: List[str] = field(default_factory=list)
    uncertainty_overconfidence_events: int = 0

    def assert_valid(self) -> None:
        """Optional strict checker for debugging and tests."""
        if self.step_count < 0:
            raise ValueError("step_count must be >= 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.task_domain not in {x.value for x in TaskDomain}:
            raise ValueError(f"Unknown task_domain: {self.task_domain}")
        if self.difficulty not in {x.value for x in DifficultyLevel}:
            raise ValueError(f"Unknown difficulty: {self.difficulty}")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Compatibility shim for OpenEnv /schema endpoint."""
        return TypeAdapter(cls).json_schema()

    def __call__(self) -> "EpisodeState":
        """
        Compatibility helper: allows both `env.state` and `env.state()`.
        Some OpenEnv examples use method-style `state()`, while this repo
        exposes `state` as a property on the environment.
        """
        return self


@dataclass
class ScenarioProfile:
    """
    Dynamic episode profile used by all packs.

    This is persisted into EpisodeState to enable deterministic replay.
    """
    scenario_id: str
    domain: str
    difficulty: str
    seed: int
    attack_profile: Dict[str, Any] = field(default_factory=dict)
    fault_profile: Dict[str, Any] = field(default_factory=dict)
    load_profile: Dict[str, Any] = field(default_factory=dict)
    compliance_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeTrace:
    """
    Structured trace row for a single step.
    """
    step_idx: int
    action_type: str
    tool_name: Optional[str]
    reward: float
    violations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    policy_decision: Optional[str] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    done: bool = False
    termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
