# Copyright (c) 2026 Agent Gauntlet Contributors
# BSD-style license

"""
Adversarial Task Generator — Feature 1.

Two-player RL loop:
  Generator (adversarial RL agent) creates failure combinations
  Solver (main agent) tries to complete tasks despite them

Generator reward = Solver failure rate on generated tasks.
This creates an arms race: Generator discovers novel failure combos,
Solver learns to handle them.

Architecture:
  AdversarialGenerator   — proposes failure injection configs
  AdversarialRecord      — tracks generator proposals + solver outcomes
  AdversarialStats       — aggregates discovery metrics for demo
"""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import DifficultyLevel, FailureType, TaskDomain
from .scenarios import (
    COMPLIANCE_POLICIES,
    DIFFICULTY_CONFIG,
    ERROR_TEMPLATES,
    InjectedFailure,
    RECOVERY_STRATEGIES,
    STAKEHOLDER_BELIEF_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Adversarial Proposal — what the generator proposes
# ---------------------------------------------------------------------------

@dataclass
class AdversarialProposal:
    """
    A failure injection config proposed by the adversarial generator.

    The generator learns which combinations break the solver most reliably.
    """
    proposal_id: str
    failure_combo: List[str]          # failure types to inject
    injection_steps: List[int]        # which steps to inject at
    cascade_enabled: bool             # whether to chain failures
    security_injected: bool           # whether to include prompt injection
    compliance_injected: bool         # whether to include policy violation
    generator_confidence: float       # generator's predicted solver failure rate
    domain: str
    difficulty: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "failure_combo": self.failure_combo,
            "injection_steps": self.injection_steps,
            "cascade_enabled": self.cascade_enabled,
            "security_injected": self.security_injected,
            "compliance_injected": self.compliance_injected,
            "generator_confidence": round(self.generator_confidence, 3),
            "domain": self.domain,
            "difficulty": self.difficulty,
        }


@dataclass
class AdversarialOutcome:
    """Result of running a solver against an adversarial proposal."""
    proposal_id: str
    solver_reward: float
    solver_failed: bool               # True if reward < threshold
    failures_detected: int
    total_failures: int
    task_completed: bool
    steps_taken: int

    @property
    def detection_rate(self) -> float:
        return self.failures_detected / max(1, self.total_failures)


# ---------------------------------------------------------------------------
# Adversarial Generator — learns which combos break solvers
# ---------------------------------------------------------------------------

class AdversarialGenerator:
    """
    Adversarial RL agent that generates failure combinations to break solvers.

    Uses a simple bandit-style learning rule:
    - Track success rate of each failure combo (success = solver failed)
    - Increase probability of combos that break solvers
    - Explore new combos with epsilon-greedy

    This is a lightweight approximation of full adversarial RL that runs
    without a separate model — suitable for hackathon demo.
    """

    # Failure types that can be combined adversarially
    _ADVERSARIAL_TYPES = [
        FailureType.API_500,
        FailureType.RATE_LIMIT_429,
        FailureType.AUTH_401,
        FailureType.CASCADING,
        FailureType.SEMANTIC_DRIFT,
        FailureType.SECURITY_BREACH,
        FailureType.COMPLIANCE_VIOLATION,
        FailureType.SLA_BREACH,
        FailureType.ADVERSARIAL_INPUT,
        FailureType.TIMEOUT,
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        # Bandit arms: combo_key → (attempts, solver_failures)
        self._combo_stats: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        self._proposals: List[AdversarialProposal] = []
        self._outcomes: List[AdversarialOutcome] = []
        self._epsilon = 0.3   # exploration rate
        self._failure_threshold = 0.2  # solver reward below this = "failed"

    def propose(
        self,
        domain: str,
        difficulty: str,
        max_steps: int,
    ) -> AdversarialProposal:
        """
        Generate an adversarial failure injection config.

        Uses epsilon-greedy: exploit known-hard combos or explore new ones.
        """
        if self._rng.random() < self._epsilon or not self._combo_stats:
            # Explore: random combo
            n_failures = self._rng.randint(2, min(5, len(self._ADVERSARIAL_TYPES)))
            combo = [
                ft.value for ft in self._rng.sample(self._ADVERSARIAL_TYPES, n_failures)
            ]
        else:
            # Exploit: pick combo with highest solver failure rate
            best_key = max(
                self._combo_stats,
                key=lambda k: (
                    self._combo_stats[k][1] / max(1, self._combo_stats[k][0])
                ),
            )
            combo = best_key.split("|")

        # Spread injection steps across episode
        available = list(range(2, max_steps - 2))
        n_steps = min(len(combo), len(available))
        injection_steps = sorted(self._rng.sample(available, n_steps))

        # Decide special injections
        security = FailureType.SECURITY_BREACH.value in combo
        compliance = FailureType.COMPLIANCE_VIOLATION.value in combo
        cascade = FailureType.CASCADING.value in combo

        # Estimate generator confidence from historical data
        combo_key = "|".join(sorted(combo))
        attempts, failures = self._combo_stats[combo_key]
        confidence = failures / max(1, attempts) if attempts > 0 else 0.5

        proposal = AdversarialProposal(
            proposal_id=str(uuid.uuid4())[:8],
            failure_combo=combo,
            injection_steps=injection_steps,
            cascade_enabled=cascade,
            security_injected=security,
            compliance_injected=compliance,
            generator_confidence=confidence,
            domain=domain,
            difficulty=difficulty,
        )
        self._proposals.append(proposal)
        return proposal

    def record_outcome(self, outcome: AdversarialOutcome) -> None:
        """Update bandit stats based on solver outcome."""
        self._outcomes.append(outcome)

        # Find the proposal
        proposal = next(
            (p for p in self._proposals if p.proposal_id == outcome.proposal_id),
            None,
        )
        if proposal is None:
            return

        combo_key = "|".join(sorted(proposal.failure_combo))
        attempts, failures = self._combo_stats[combo_key]
        self._combo_stats[combo_key] = (
            attempts + 1,
            failures + (1 if outcome.solver_failed else 0),
        )

    def build_failure_schedule(
        self,
        proposal: AdversarialProposal,
        available_tools: List[str],
        rng: random.Random,
    ) -> List[InjectedFailure]:
        """
        Convert an adversarial proposal into a concrete InjectedFailure schedule.
        Uses existing ERROR_TEMPLATES — no hardcoded messages.
        """
        failures = []
        cascade_steps: List[int] = []

        for i, (ft_value, step) in enumerate(
            zip(proposal.failure_combo, proposal.injection_steps)
        ):
            try:
                ft = FailureType(ft_value)
            except ValueError:
                continue

            error_info = rng.choice(
                ERROR_TEMPLATES.get(ft, [{"msg": "Adversarial failure", "code": 500}])
            )

            # Cascade: first cascading failure triggers subsequent steps
            is_cascade = ft == FailureType.CASCADING and proposal.cascade_enabled
            cascade_affects: List[int] = []
            if is_cascade and not cascade_steps:
                n_cascade = rng.randint(2, 4)
                cascade_affects = list(range(step + 1, min(step + n_cascade + 1, 200)))
                cascade_steps.extend(cascade_affects)

            # Compliance metadata
            compliance_policy = None
            forbidden_action = None
            if ft == FailureType.COMPLIANCE_VIOLATION:
                policy = rng.choice(COMPLIANCE_POLICIES)
                compliance_policy = policy["name"]
                forbidden_action = rng.choice(policy["forbidden_actions"])

            # ToM scenario for semantic drift / security / cascading
            tom_scenario = None
            if ft in (FailureType.SEMANTIC_DRIFT, FailureType.SECURITY_BREACH, FailureType.CASCADING):
                tom_scenario = rng.choice(STAKEHOLDER_BELIEF_SCENARIOS)

            failures.append(InjectedFailure(
                step=step,
                failure_type=ft,
                tool_name=rng.choice(available_tools),
                error_message=error_info["msg"],
                status_code=error_info["code"],
                is_cascade_trigger=is_cascade,
                cascade_affects_steps=cascade_affects,
                requires_recovery=(ft not in (FailureType.CONTEXT_PRESSURE, FailureType.SLA_BREACH)),
                correct_recovery=RECOVERY_STRATEGIES.get(ft, "retry"),
                compliance_policy=compliance_policy,
                forbidden_action=forbidden_action,
                tom_scenario=tom_scenario,
            ))

        return failures

    def top_breaking_combos(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the failure combos that break solvers most reliably."""
        ranked = sorted(
            [
                {
                    "combo": k.split("|"),
                    "attempts": v[0],
                    "solver_failures": v[1],
                    "failure_rate": round(v[1] / max(1, v[0]), 3),
                }
                for k, v in self._combo_stats.items()
                if v[0] > 0
            ],
            key=lambda x: x["failure_rate"],
            reverse=True,
        )
        return ranked[:top_k]

    def stats(self) -> Dict[str, Any]:
        """Full adversarial generator stats for /adversarial/stats endpoint."""
        total_proposals = len(self._proposals)
        total_outcomes = len(self._outcomes)
        solver_failures = sum(1 for o in self._outcomes if o.solver_failed)

        return {
            "total_proposals": total_proposals,
            "total_outcomes": total_outcomes,
            "solver_failure_rate": round(solver_failures / max(1, total_outcomes), 3),
            "top_breaking_combos": self.top_breaking_combos(),
            "unique_combos_discovered": len(self._combo_stats),
            "epsilon": self._epsilon,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — shared across environment instances
# ---------------------------------------------------------------------------

_GLOBAL_GENERATOR = AdversarialGenerator()


def get_global_generator() -> AdversarialGenerator:
    return _GLOBAL_GENERATOR
