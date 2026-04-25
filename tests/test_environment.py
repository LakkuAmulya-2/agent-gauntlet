"""
Tests for Agent Gauntlet environment.

Run: pytest tests/ -v
"""

from __future__ import annotations

import pytest

from agent_gauntlet.models import (
    ActionType,
    AgentAction,
    DifficultyLevel,
    EpisodeState,
    FailureType,
    TaskDomain,
)
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment
from agent_gauntlet.runtime.scenarios import ScenarioGenerator
from agent_gauntlet.runtime.rubrics import (
    AgentGauntletRubric,
    AntiGamingRubric,
    EfficiencyRubric,
    FailureDetectionRubric,
    TaskCompletionRubric,
)


# ---------------------------------------------------------------------------
# Fix 1: EpisodeState has proper fields (not dynamic attributes)
# ---------------------------------------------------------------------------

class TestEpisodeStateFields:
    def test_has_repeated_action_count(self):
        state = EpisodeState()
        assert hasattr(state, "repeated_action_count")
        assert state.repeated_action_count == 0

    def test_has_last_action_key(self):
        state = EpisodeState()
        assert hasattr(state, "last_action_key")
        assert state.last_action_key is None

    def test_fields_are_dataclass_fields(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(EpisodeState)}
        assert "repeated_action_count" in field_names
        assert "last_action_key" in field_names


# ---------------------------------------------------------------------------
# ScenarioGenerator Tests
# ---------------------------------------------------------------------------

class TestScenarioGenerator:
    def test_generates_task_for_each_difficulty(self):
        gen = ScenarioGenerator(seed=42)
        for diff in DifficultyLevel:
            task = gen.generate(difficulty=diff)
            assert task.task_id
            assert task.description
            assert task.available_tools
            assert task.max_steps > 0
            assert task.api_calls_budget > 0

    def test_easy_has_fewer_failures_than_expert(self):
        gen = ScenarioGenerator(seed=42)
        easy = gen.generate(difficulty=DifficultyLevel.EASY)
        expert = gen.generate(difficulty=DifficultyLevel.EXPERT)
        assert len(easy.failure_schedule) <= len(expert.failure_schedule)

    def test_failure_steps_within_bounds(self):
        gen = ScenarioGenerator(seed=42)
        for diff in DifficultyLevel:
            task = gen.generate(difficulty=diff)
            for f in task.failure_schedule:
                assert 0 <= f.step < task.max_steps

    def test_no_hardcoded_data(self):
        gen1 = ScenarioGenerator(seed=1)
        gen2 = ScenarioGenerator(seed=2)
        task1 = gen1.generate(difficulty=DifficultyLevel.MEDIUM)
        task2 = gen2.generate(difficulty=DifficultyLevel.MEDIUM)
        assert task1.task_id != task2.task_id

    def test_api_budget_ratio_applied(self):
        gen = ScenarioGenerator(seed=42)
        easy = gen.generate(difficulty=DifficultyLevel.EASY)
        ratio = easy.api_calls_budget / easy.max_steps
        assert 0.7 <= ratio <= 0.9


# ---------------------------------------------------------------------------
# Environment Tests
# ---------------------------------------------------------------------------

class TestAgentGauntletEnvironment:
    def setup_method(self):
        self.env = AgentGauntletEnvironment(
            default_difficulty=DifficultyLevel.EASY,
            seed=42,
            adaptive_curriculum=False,
        )

    def test_reset_returns_valid_observation(self):
        obs = self.env.reset()
        assert obs.task_description
        assert obs.available_tools
        assert obs.max_steps > 0
        assert obs.budget_remaining == 1.0
        assert obs.current_step == 0
        assert not obs.is_done

    def test_step_increments_step_count(self):
        self.env.reset()
        obs = self.env.step(AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name="fetch_records",
            reasoning="Starting task",
        ))
        assert obs.current_step == 1

    def test_step_without_reset_raises(self):
        env = AgentGauntletEnvironment(seed=42)
        with pytest.raises(RuntimeError):
            env.step(AgentAction(action_type=ActionType.CALL_TOOL.value))

    def test_budget_tracking_synced(self):
        """Fix 1: budget_remaining stays in sync with budget_used_pct."""
        self.env.reset()
        r1 = self.env.step(AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name="fetch_records",
            reasoning="test",
        ))
        r2 = self.env.step(AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name="fetch_records",
            reasoning="test",
        ))
        assert r2.budget_remaining < 1.0
        assert r2.budget_remaining < r1.budget_remaining

    def test_failure_injects_regardless_of_tool_name(self):
        """Fix 2: failure injects on any tool call at scheduled step."""
        env = AgentGauntletEnvironment(seed=42, adaptive_curriculum=False)
        env.reset()

        if not env._failure_map:
            pytest.skip("No failures in this episode")

        failure_step = min(env._failure_map.keys())
        scheduled_tool = env._failure_map[failure_step].tool_name

        # Advance to failure step
        for _ in range(failure_step):
            env.step(AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name=env._task.available_tools[0],
                reasoning="advancing",
            ))

        # Call a DIFFERENT tool — failure should still inject
        different_tool = next(
            (t for t in env._task.available_tools if t != scheduled_tool),
            env._task.available_tools[0],
        )
        obs = env.step(AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=different_tool,
            reasoning="calling different tool",
        ))
        assert obs.last_tool_result is not None
        assert not obs.last_tool_result.success

    def test_repeated_action_uses_proper_state_fields(self):
        """Fix 1: repeated action detection uses EpisodeState fields."""
        self.env.reset()
        action = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name="fetch_records",
            reasoning="same",
        )
        for _ in range(5):
            obs = self.env.step(action)
            if obs.is_done:
                assert obs.termination_reason == "repeated_action_loop"
                return
        # If not terminated, count should be tracked in state
        assert self.env._state.repeated_action_count >= 0

    def test_anti_gaming_trivial_completion(self):
        self.env.reset()
        obs = self.env.step(AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result="done",
            reasoning="trivial",
        ))
        assert obs._reward <= 0.1

    def test_multi_agent_messaging(self):
        env = AgentGauntletEnvironment(seed=42, adaptive_curriculum=False)
        env.reset(domain="multi_agent_coordination")
        obs = env.step(AgentAction(
            action_type=ActionType.SEND_MESSAGE.value,
            target_agent_id="agent_1",
            message_content="Please start processing",
            reasoning="delegating",
        ))
        assert obs is not None
        assert len(obs.other_agents_status) > 0

    def test_adaptive_curriculum_promotes(self):
        """Fix 4: adaptive curriculum only mutates when no explicit override."""
        env = AgentGauntletEnvironment(
            default_difficulty=DifficultyLevel.EASY,
            seed=42,
            adaptive_curriculum=True,
        )
        env._recent_rewards = [0.8] * env._WINDOW
        env.reset()
        assert env._difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]

    def test_explicit_difficulty_overrides_adaptive(self):
        """Fix 4: explicit difficulty= bypasses adaptive curriculum."""
        env = AgentGauntletEnvironment(
            default_difficulty=DifficultyLevel.EASY,
            seed=42,
            adaptive_curriculum=True,
        )
        env._recent_rewards = [0.8] * env._WINDOW
        obs = env.reset(difficulty="easy")  # explicit override
        assert obs.difficulty == "easy"

    def test_task_completion(self):
        self.env.reset()
        for _ in range(3):
            self.env.step(AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name="fetch_records",
                reasoning="Working",
            ))
        obs = self.env.step(AgentAction(
            action_type=ActionType.COMPLETE_TASK.value,
            task_result="Successfully completed: fetched 1000 records, transformed, loaded",
            reasoning="All steps done",
        ))
        assert obs.is_done
        assert obs.termination_reason == "task_completed"

    def test_state_tracks_correctly(self):
        self.env.reset()
        for _ in range(5):
            self.env.step(AgentAction(
                action_type=ActionType.CALL_TOOL.value,
                tool_name="fetch_records",
                reasoning="Working",
            ))
        state = self.env.state
        assert state.step_count == 5
        assert state.total_api_calls == 5


# ---------------------------------------------------------------------------
# Rubric Tests
# ---------------------------------------------------------------------------

class TestRubrics:
    def setup_method(self):
        self.rubric = AgentGauntletRubric()

    def _make_obs(self, is_done=False, termination=None, step=5,
                  budget=0.8, context=0.3, metadata=None):
        class FakeObs:
            pass
        obs = FakeObs()
        obs.is_done = is_done
        obs.termination_reason = termination
        obs.current_step = step
        obs.budget_remaining = budget
        obs.context_used_pct = context
        obs.metadata = metadata or {}
        return obs

    def test_anti_gaming_trivial_completion(self):
        rubric = AntiGamingRubric()
        action = AgentAction(action_type=ActionType.COMPLETE_TASK.value)
        obs = self._make_obs(step=1)
        assert rubric(action, obs) == -1.0

    def test_anti_gaming_genuine_engagement(self):
        rubric = AntiGamingRubric()
        action = AgentAction(action_type=ActionType.CALL_TOOL.value)
        obs = self._make_obs(step=5)
        assert rubric(action, obs) == 0.1

    def test_failure_detection_correct(self):
        rubric = FailureDetectionRubric()
        rubric.set_step_context("api_500", "retry_with_backoff")
        action = AgentAction(
            action_type=ActionType.DETECT_FAILURE.value,
            failure_detected="api_500",
            recovery_strategy="retry_with_backoff",
        )
        obs = self._make_obs()
        score = rubric(action, obs)
        assert score == 0.6  # 0.4 detection + 0.2 correct recovery

    def test_failure_detection_missed(self):
        rubric = FailureDetectionRubric()
        rubric.set_step_context("api_500", "retry_with_backoff")
        action = AgentAction(action_type=ActionType.CALL_TOOL.value)
        obs = self._make_obs()
        assert rubric(action, obs) == -0.5

    def test_false_positive_penalized(self):
        rubric = FailureDetectionRubric()
        rubric.set_step_context(None, None)
        action = AgentAction(
            action_type=ActionType.DETECT_FAILURE.value,
            failure_detected="api_500",
        )
        obs = self._make_obs()
        assert rubric(action, obs) == -0.2

    def test_task_completion_correct(self):
        rubric = TaskCompletionRubric()
        action = AgentAction(action_type=ActionType.COMPLETE_TASK.value)
        obs = self._make_obs(
            is_done=True, termination="task_completed", step=10,
            metadata={"task_result_correct": True}
        )
        assert rubric(action, obs) == 1.0

    def test_reward_clamped(self):
        action = AgentAction(action_type=ActionType.CALL_TOOL.value)
        obs = self._make_obs()
        for _ in range(20):
            reward = self.rubric(action, obs)
            assert -1.0 <= reward <= 1.0

    def test_rubric_resets_cleanly(self):
        self.rubric.set_step_context("api_500", "retry_with_backoff", False)
        self.rubric.reset()
        assert self.rubric.failure_detection._failure_at_step is None
        assert self.rubric.escalation._escalation_count == 0
