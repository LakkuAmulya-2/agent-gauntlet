# Copyright (c) 2026 Agent Gauntlet Contributors
# BSD-style license

"""
Kaizen Kernel — Continuous Self-Improvement Engine for Agent Gauntlet.

"Kaizen" (改善) = change for the better.

This module converts Agent Gauntlet from a static benchmark into an
adaptive evolution engine:

  Episode N ends
    → KaizenKernel analyzes failures + traces
    → Updates SkillProfile (per-failure-type mastery scores)
    → Adjusts curriculum weights (more practice on weak skills)
    → Generates targeted SFT examples from high-quality traces
    → Episode N+1 starts with targeted failure injection

Architecture:
  KaizenKernel          — orchestrates the loop
  SkillProfile          — tracks mastery per failure type + domain
  CurriculumScheduler   — decides next episode's failure mix
  TraceMemory           — stores + retrieves high-quality traces
  EvolutionLineage      — records agent version history for demo
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import DifficultyLevel, FailureType, TaskDomain


# ---------------------------------------------------------------------------
# Skill Profile — per-failure-type mastery tracking
# ---------------------------------------------------------------------------

@dataclass
class SkillScore:
    """Mastery score for one skill (failure type or domain)."""
    skill_id: str
    attempts: int = 0
    successes: int = 0
    recent_rewards: List[float] = field(default_factory=list)
    _window: int = field(default=10, repr=False)

    @property
    def mastery(self) -> float:
        """0.0 (no mastery) → 1.0 (fully mastered). Requires ≥3 attempts."""
        if self.attempts < 3:
            return 0.0
        recent = self.recent_rewards[-self._window:]
        return round(sum(recent) / len(recent), 4) if recent else 0.0

    @property
    def is_weak(self) -> bool:
        return self.attempts >= 3 and self.mastery < 0.35

    @property
    def is_mastered(self) -> bool:
        return self.attempts >= 5 and self.mastery >= 0.65

    def record(self, reward: float, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self._window * 2:
            self.recent_rewards = self.recent_rewards[-self._window:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "attempts": self.attempts,
            "successes": self.successes,
            "mastery": self.mastery,
            "is_weak": self.is_weak,
            "is_mastered": self.is_mastered,
        }


class SkillProfile:
    """
    Tracks mastery across all failure types and task domains.

    Used by CurriculumScheduler to decide what to practice next.
    """

    def __init__(self) -> None:
        self._failure_skills: Dict[str, SkillScore] = {
            ft.value: SkillScore(skill_id=ft.value)
            for ft in FailureType
            if ft != FailureType.NONE
        }
        self._domain_skills: Dict[str, SkillScore] = {
            td.value: SkillScore(skill_id=td.value)
            for td in TaskDomain
        }

    def record_episode(
        self,
        failure_types_encountered: List[str],
        failures_detected: int,
        total_failures: int,
        domain: str,
        episode_reward: float,
        task_completed: bool,
    ) -> None:
        """Update skill scores after an episode ends."""
        detection_rate = failures_detected / max(1, total_failures)
        success = task_completed and detection_rate >= 0.5

        for ft in failure_types_encountered:
            if ft in self._failure_skills:
                self._failure_skills[ft].record(episode_reward, success)

        if domain in self._domain_skills:
            self._domain_skills[domain].record(episode_reward, task_completed)

    @property
    def weak_failures(self) -> List[str]:
        """Failure types the agent struggles with most."""
        return [
            s.skill_id for s in self._failure_skills.values()
            if s.is_weak
        ]

    @property
    def mastered_failures(self) -> List[str]:
        """Failure types the agent handles reliably."""
        return [
            s.skill_id for s in self._failure_skills.values()
            if s.is_mastered
        ]

    @property
    def weak_domains(self) -> List[str]:
        """Task domains the agent struggles with."""
        return [
            s.skill_id for s in self._domain_skills.values()
            if s.is_weak
        ]

    def summary(self) -> Dict[str, Any]:
        return {
            "failure_skills": {k: v.to_dict() for k, v in self._failure_skills.items()},
            "domain_skills": {k: v.to_dict() for k, v in self._domain_skills.items()},
            "weak_failures": self.weak_failures,
            "mastered_failures": self.mastered_failures,
            "weak_domains": self.weak_domains,
        }


# ---------------------------------------------------------------------------
# Trace Memory — stores high-quality diagnostic traces
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    """One stored diagnostic trace with context."""
    episode_id: str
    failure_type: str
    domain: str
    difficulty: str
    trace_text: str
    quality_score: float
    episode_reward: float
    timestamp: float = field(default_factory=time.time)

    def to_sft_example(self) -> Dict[str, Any]:
        """Convert to SFT training example format."""
        return {
            "role": "assistant",
            "content": json.dumps({
                "action_type": "generate_trace",
                "diagnostic_trace": self.trace_text,
                "reasoning": f"Generating trace after {self.failure_type} failure",
            }),
            "metadata": {
                "failure_type": self.failure_type,
                "domain": self.domain,
                "quality_score": self.quality_score,
            },
        }


class TraceMemory:
    """
    Stores high-quality diagnostic traces across episodes.

    High-quality traces (score ≥ 0.2) are kept as training examples
    for the next SFT round — this is the genuine self-improvement loop.
    """

    def __init__(self, max_traces: int = 500, quality_threshold: float = 0.15) -> None:
        self._traces: List[TraceRecord] = []
        self._max = max_traces
        self._threshold = quality_threshold

    def add(
        self,
        episode_id: str,
        failure_type: str,
        domain: str,
        difficulty: str,
        trace_text: str,
        quality_score: float,
        episode_reward: float,
    ) -> bool:
        """Add trace if quality meets threshold. Returns True if stored."""
        if quality_score < self._threshold:
            return False
        record = TraceRecord(
            episode_id=episode_id,
            failure_type=failure_type,
            domain=domain,
            difficulty=difficulty,
            trace_text=trace_text,
            quality_score=quality_score,
            episode_reward=episode_reward,
        )
        self._traces.append(record)
        # Keep only top-quality traces when at capacity
        if len(self._traces) > self._max:
            self._traces.sort(key=lambda t: t.quality_score, reverse=True)
            self._traces = self._traces[: self._max]
        return True

    def get_for_failure(self, failure_type: str, top_k: int = 3) -> List[TraceRecord]:
        """Retrieve best traces for a specific failure type."""
        relevant = [t for t in self._traces if t.failure_type == failure_type]
        relevant.sort(key=lambda t: t.quality_score, reverse=True)
        return relevant[:top_k]

    def export_sft_dataset(self) -> List[Dict[str, Any]]:
        """Export all stored traces as SFT training examples."""
        return [t.to_sft_example() for t in self._traces]

    @property
    def size(self) -> int:
        return len(self._traces)

    def summary(self) -> Dict[str, Any]:
        if not self._traces:
            return {"total": 0, "by_failure_type": {}}
        by_type: Dict[str, int] = defaultdict(int)
        for t in self._traces:
            by_type[t.failure_type] += 1
        return {
            "total": len(self._traces),
            "avg_quality": round(sum(t.quality_score for t in self._traces) / len(self._traces), 4),
            "by_failure_type": dict(by_type),
        }


# ---------------------------------------------------------------------------
# Curriculum Scheduler — decides next episode's failure mix
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Adaptive curriculum based on SkillProfile.

    Logic:
    - Weak skills → inject those failure types more frequently
    - Mastered skills → reduce frequency, introduce harder variants
    - Unknown skills → explore with uniform probability
    - Mixes difficulty levels based on overall performance
    """

    # Base injection probabilities per difficulty (from DIFFICULTY_CONFIG)
    _BASE_SECURITY_PROB = {
        "easy": 0.0, "medium": 0.10, "hard": 0.25, "expert": 0.40
    }
    _BASE_COMPLIANCE_PROB = {
        "easy": 0.0, "medium": 0.10, "hard": 0.20, "expert": 0.35
    }

    def __init__(self, skill_profile: SkillProfile) -> None:
        self._profile = skill_profile
        self._episode_count = 0

    def next_episode_config(
        self,
        current_difficulty: str,
        recent_avg_reward: float,
    ) -> Dict[str, Any]:
        """
        Returns config overrides for the next episode.

        Returns dict with keys:
          difficulty: str
          domain: Optional[str]  — targeted domain if weak
          boost_failure_types: List[str]  — inject these more
          use_harder_variant: bool
        """
        self._episode_count += 1
        weak = self._profile.weak_failures
        mastered = self._profile.mastered_failures
        weak_domains = self._profile.weak_domains

        # Difficulty selection — mix based on performance
        difficulty = self._select_difficulty(current_difficulty, recent_avg_reward)

        # Domain targeting — focus on weak domains 40% of the time
        domain = None
        if weak_domains and self._episode_count % 3 == 0:
            # Rotate through weak domains
            domain = weak_domains[self._episode_count % len(weak_domains)]

        # Failure type boosting — target weak skills
        boost = []
        if weak:
            # Boost up to 2 weak failure types per episode
            boost = weak[: min(2, len(weak))]

        # Use harder variant if agent is doing well overall
        use_harder = recent_avg_reward >= 0.60 and len(mastered) >= 3

        return {
            "difficulty": difficulty,
            "domain": domain,
            "boost_failure_types": boost,
            "use_harder_variant": use_harder,
            "weak_skills": weak,
            "mastered_skills": mastered,
        }

    def _select_difficulty(self, current: str, avg_reward: float) -> str:
        """
        Mixed curriculum: don't always use the same difficulty.

        - 60% current difficulty
        - 20% one level harder (challenge)
        - 20% one level easier (consolidation)
        """
        order = ["easy", "medium", "hard", "expert"]
        idx = order.index(current) if current in order else 0

        import random
        roll = random.random()
        if roll < 0.60:
            return order[idx]
        elif roll < 0.80:
            return order[min(idx + 1, len(order) - 1)]
        else:
            return order[max(idx - 1, 0)]


# ---------------------------------------------------------------------------
# Evolution Lineage — records agent version history
# ---------------------------------------------------------------------------

@dataclass
class AgentVersion:
    """Snapshot of agent performance at a point in time."""
    version: int
    episode: int
    avg_reward: float
    difficulty: str
    weak_skills: List[str]
    mastered_skills: List[str]
    skill_breakdown: Dict[str, float]  # skill_id → mastery score
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvolutionLineage:
    """
    Records agent version history for the "Learning Curve Engine" demo.

    Judges can see: Agent v1 vs v5 vs v10 — improvement graph.
    """

    def __init__(self) -> None:
        self._versions: List[AgentVersion] = []
        self._snapshot_interval = 10  # snapshot every N episodes

    def maybe_snapshot(
        self,
        episode: int,
        avg_reward: float,
        difficulty: str,
        skill_profile: SkillProfile,
    ) -> Optional[AgentVersion]:
        """Take a snapshot if interval reached. Returns version or None."""
        if episode % self._snapshot_interval != 0:
            return None

        version_num = len(self._versions) + 1
        skill_summary = skill_profile.summary()
        breakdown = {
            k: v["mastery"]
            for k, v in skill_summary["failure_skills"].items()
            if v["attempts"] > 0
        }

        v = AgentVersion(
            version=version_num,
            episode=episode,
            avg_reward=round(avg_reward, 4),
            difficulty=difficulty,
            weak_skills=skill_summary["weak_failures"],
            mastered_skills=skill_summary["mastered_failures"],
            skill_breakdown=breakdown,
        )
        self._versions.append(v)
        return v

    @property
    def versions(self) -> List[AgentVersion]:
        return list(self._versions)

    def learning_curve(self) -> Dict[str, Any]:
        """Returns data for the learning curve plot."""
        if not self._versions:
            return {"versions": [], "rewards": [], "difficulties": []}
        return {
            "versions": [v.version for v in self._versions],
            "episodes": [v.episode for v in self._versions],
            "rewards": [v.avg_reward for v in self._versions],
            "difficulties": [v.difficulty for v in self._versions],
            "mastered_count": [len(v.mastered_skills) for v in self._versions],
            "weak_count": [len(v.weak_skills) for v in self._versions],
        }

    def skill_growth_table(self) -> List[Dict[str, Any]]:
        """
        Returns before/after skill table for demo.

        Example:
          Skill          | v1    | v5    | v10
          rate_limit_429 | 0.12  | 0.45  | 0.78
          security_breach| 0.05  | 0.31  | 0.62
        """
        if len(self._versions) < 2:
            return []

        first = self._versions[0]
        last = self._versions[-1]
        all_skills = set(first.skill_breakdown) | set(last.skill_breakdown)

        rows = []
        for skill in sorted(all_skills):
            before = first.skill_breakdown.get(skill, 0.0)
            after = last.skill_breakdown.get(skill, 0.0)
            rows.append({
                "skill": skill,
                "before": round(before, 3),
                "after": round(after, 3),
                "improvement": round(after - before, 3),
                "improved": after > before + 0.05,
            })
        rows.sort(key=lambda r: r["improvement"], reverse=True)
        return rows


# ---------------------------------------------------------------------------
# Kaizen Kernel — main orchestrator
# ---------------------------------------------------------------------------

class KaizenKernel:
    """
    Adaptive Agent Evolution Engine.

    Converts Agent Gauntlet from static benchmark → evolution engine:

    1. SkillProfile tracks mastery per failure type + domain
    2. CurriculumScheduler targets weak skills in next episode
    3. TraceMemory stores high-quality traces as future training data
    4. EvolutionLineage records agent version history for demo

    Usage:
        kernel = KaizenKernel()

        # After each episode:
        config = kernel.on_episode_end(
            episode_id=...,
            episode_reward=...,
            metadata=obs.metadata,
            traces=state.diagnostic_traces,
            trace_quality_scores=state.trace_quality_scores,
            current_difficulty=...,
        )

        # Use config for next episode:
        env.reset(
            difficulty=config["difficulty"],
            domain=config["domain"],
            use_harder_variant=config["use_harder_variant"],
        )
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._skill_profile = SkillProfile()
        self._trace_memory = TraceMemory()
        self._lineage = EvolutionLineage()
        self._scheduler = CurriculumScheduler(self._skill_profile)
        self._episode_count = 0
        self._recent_rewards: List[float] = []
        self._persist_path = persist_path

    def on_episode_end(
        self,
        episode_id: str,
        episode_reward: float,
        metadata: Dict[str, Any],
        traces: List[str],
        trace_quality_scores: List[float],
        current_difficulty: str,
        domain: str = "data_pipeline",
    ) -> Dict[str, Any]:
        """
        Process episode results and return config for next episode.

        Called by AgentGauntletEnvironment at end of each episode.
        """
        self._episode_count += 1
        self._recent_rewards.append(episode_reward)
        if len(self._recent_rewards) > 20:
            self._recent_rewards = self._recent_rewards[-20:]

        # Extract episode metadata
        injected = metadata.get("injected_failures", [])
        failure_types = [f.get("type", "") for f in injected] if isinstance(injected, list) else []
        failures_detected = metadata.get("failures_detected_correctly", 0)
        total_failures = metadata.get("total_injected_failures", 0)
        task_completed = metadata.get("task_result_correct", False)

        # Update skill profile
        self._skill_profile.record_episode(
            failure_types_encountered=failure_types,
            failures_detected=failures_detected,
            total_failures=total_failures,
            domain=domain,
            episode_reward=episode_reward,
            task_completed=task_completed,
        )

        # Store high-quality traces in memory
        for i, trace in enumerate(traces):
            quality = trace_quality_scores[i] if i < len(trace_quality_scores) else 0.0
            # Determine which failure type this trace is about
            trace_lower = trace.lower()
            matched_ft = "unknown"
            for ft in FailureType:
                if ft.value.replace("_", " ") in trace_lower or ft.value in trace_lower:
                    matched_ft = ft.value
                    break
            self._trace_memory.add(
                episode_id=episode_id,
                failure_type=matched_ft,
                domain=domain,
                difficulty=current_difficulty,
                trace_text=trace,
                quality_score=quality,
                episode_reward=episode_reward,
            )

        # Snapshot lineage
        avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)
        self._lineage.maybe_snapshot(
            episode=self._episode_count,
            avg_reward=avg_reward,
            difficulty=current_difficulty,
            skill_profile=self._skill_profile,
        )

        # Get next episode config from scheduler
        next_config = self._scheduler.next_episode_config(
            current_difficulty=current_difficulty,
            recent_avg_reward=avg_reward,
        )

        # Persist state if path configured
        if self._persist_path:
            self._save(self._persist_path)

        return next_config

    def get_relevant_traces(self, failure_type: str, top_k: int = 3) -> List[str]:
        """
        Retrieve best traces for a failure type.

        Used to inject past lessons into the agent's context:
        "In a previous episode, you handled this failure by..."
        """
        records = self._trace_memory.get_for_failure(failure_type, top_k)
        return [r.trace_text for r in records]

    def report(self) -> Dict[str, Any]:
        """Full Kaizen report — exposed via /kaizen/report API endpoint."""
        avg_reward = (
            sum(self._recent_rewards) / len(self._recent_rewards)
            if self._recent_rewards else 0.0
        )
        return {
            "episode_count": self._episode_count,
            "avg_reward_recent": round(avg_reward, 4),
            "skill_profile": self._skill_profile.summary(),
            "trace_memory": self._trace_memory.summary(),
            "learning_curve": self._lineage.learning_curve(),
            "skill_growth_table": self._lineage.skill_growth_table(),
            "versions": [v.to_dict() for v in self._lineage.versions],
        }

    def export_sft_dataset(self) -> List[Dict[str, Any]]:
        """Export trace memory as SFT training examples for next round."""
        return self._trace_memory.export_sft_dataset()

    def _save(self, path: str) -> None:
        """Persist Kaizen state to JSON."""
        try:
            Path(path).write_text(
                json.dumps(self.report(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass  # non-critical

    @property
    def skill_profile(self) -> SkillProfile:
        return self._skill_profile

    @property
    def trace_memory(self) -> TraceMemory:
        return self._trace_memory

    @property
    def lineage(self) -> EvolutionLineage:
        return self._lineage
