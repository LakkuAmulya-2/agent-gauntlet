from __future__ import annotations

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def test_reset_includes_dynamic_scenario_profile():
    env = AgentGauntletEnvironment(seed=123, adaptive_curriculum=False)
    obs = env.reset(difficulty="hard", seed=123)
    profile = env.state.scenario_profile
    perturb = env.state.perturbation_profile
    assert profile["scenario_id"] == obs.episode_id
    assert "attack_profile" in perturb
    assert "fault_profile" in perturb
    assert "load_profile" in perturb


def test_step_has_verifier_evidence_and_trace():
    env = AgentGauntletEnvironment(seed=123, adaptive_curriculum=False)
    obs = env.reset(difficulty="medium", seed=123)
    act = AgentAction(
        action_type=ActionType.CALL_TOOL.value,
        tool_name=obs.available_tools[0],
        idempotency_key=f"{obs.episode_id}:0",
        reasoning="Execute with evidence trace",
    )
    nxt = env.step(act)
    assert isinstance(nxt.verifier_evidence, list)
    assert len(nxt.verifier_evidence) >= 5
    assert len(env.state.episode_trace) >= 5


def test_done_observation_has_pack_metrics():
    env = AgentGauntletEnvironment(seed=9, adaptive_curriculum=False)
    obs = env.reset(difficulty="easy", seed=9)
    for i in range(6):
        act = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[i % len(obs.available_tools)],
            idempotency_key=f"{obs.episode_id}:{i}",
            reasoning="progress",
        )
        obs = env.step(act)
        if obs.is_done:
            break
    if not obs.is_done:
        obs = env.step(
            AgentAction(
                action_type=ActionType.COMPLETE_TASK.value,
                task_result="completed with policy checks and recovery",
                reasoning="finish",
            )
        )
    assert obs.is_done
    assert "pack_metrics" in obs.metadata
