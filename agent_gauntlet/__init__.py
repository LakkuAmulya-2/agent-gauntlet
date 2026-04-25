# Copyright (c) 2026 Agent Gauntlet Contributors
# BSD-style license

"""
Agent Gauntlet — RL Environment for Training Production-Ready AI Agents

88% of enterprise AI agents fail when moved from demo to production.
This environment trains LLMs to survive real production failure conditions:
- Tool/API failures (500s, rate limits, auth errors)
- Cascading failures across multi-step workflows
- Context pressure over long-horizon tasks
- Adversarial inputs and malformed data
- Resource/budget overruns

Usage:
    from agent_gauntlet import AgentGauntletEnv, AgentAction

    with AgentGauntletEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        print(result.observation.task_description)

        result = env.step(AgentAction(
            action_type="call_tool",
            tool_name="fetch_data",
            tool_args={"endpoint": "/api/users"},
            reasoning="Need to fetch user data to start the pipeline"
        ))
        print(result.reward)
"""

from .client import AgentGauntletEnv
from .models import AgentAction, TaskObservation, EpisodeState

__all__ = [
    "AgentGauntletEnv",
    "AgentAction",
    "TaskObservation",
    "EpisodeState",
]
