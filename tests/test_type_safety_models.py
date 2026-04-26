from __future__ import annotations

import pytest

from agent_gauntlet.models import ActionType, AgentAction, EpisodeState, TaskObservation
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def test_agent_action_accepts_enum_and_normalizes_to_string():
    action = AgentAction(action_type=ActionType.CALL_TOOL, reasoning="ok")
    assert action.action_type == ActionType.CALL_TOOL.value


def test_agent_action_rejects_unknown_action_type():
    with pytest.raises(ValueError):
        AgentAction(action_type="unknown_action", reasoning="bad")


def test_agent_action_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        AgentAction(action_type=ActionType.CALL_TOOL.value, reasoning="bad", confidence_score=1.2)


def test_observation_and_state_strict_validation_helpers():
    obs = TaskObservation(task_description="x")
    obs.assert_valid()

    state = EpisodeState(task_id="x")
    state.assert_valid()


def test_environment_state_supports_property_and_method_style():
    env = AgentGauntletEnvironment(adaptive_curriculum=False)
    env.reset()
    state_prop = env.state
    state_call = env.state()
    assert state_prop is state_call
