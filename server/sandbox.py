from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from time import time
from typing import Dict, List, Optional

from agent_gauntlet.models import ActionType, AgentAction
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment


def replay_episode(task_seed: int, difficulty: str, domain: Optional[str], responses: List[str]) -> Dict:
    env = AgentGauntletEnvironment(seed=task_seed, adaptive_curriculum=False)
    obs = env.reset(seed=task_seed, difficulty=difficulty, domain=domain)
    turns = []
    cumulative_reward = 0.0
    for i in range(min(obs.max_steps, len(responses))):
        action = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[i % len(obs.available_tools)],
            reasoning=responses[i],
            confidence_score=0.6,
            idempotency_key=f"{obs.episode_id}:{i}",
        )
        obs = env.step(action)
        reward = float(getattr(obs, "_reward", 0.0))
        cumulative_reward += reward
        turns.append(
            {
                "turn": i + 1,
                "reward": reward,
                "done": bool(obs.is_done),
                "termination_reason": obs.termination_reason,
                "violations": sum(len(ev.get("violations", [])) for ev in obs.verifier_evidence),
            }
        )
        if obs.is_done:
            break

    return {
        "episode_id": obs.episode_id,
        "difficulty": difficulty,
        "domain": obs.task_domain,
        "turns": turns,
        "cumulative_reward": cumulative_reward,
        "episode_survived": obs.termination_reason == "completed",
    }


def redteam_episode(task_seed: int, difficulty: str, domain: Optional[str], attack_messages: List[str]) -> Dict:
    env = AgentGauntletEnvironment(seed=task_seed, adaptive_curriculum=False)
    obs = env.reset(seed=task_seed, difficulty=difficulty, domain=domain)
    total_violations = 0
    for i in range(min(obs.max_steps, len(attack_messages))):
        action = AgentAction(
            action_type=ActionType.CALL_TOOL.value,
            tool_name=obs.available_tools[i % len(obs.available_tools)],
            tool_args={"payload": attack_messages[i]},
            reasoning="redteam_probe",
            confidence_score=0.99,
        )
        obs = env.step(action)
        total_violations += sum(len(ev.get("violations", [])) for ev in obs.verifier_evidence)
        if obs.is_done:
            break

    return {
        "episode_id": obs.episode_id,
        "difficulty": difficulty,
        "domain": obs.task_domain,
        "total_violations": total_violations,
        "termination_reason": obs.termination_reason,
    }


@dataclass
class ToolExecutionRecord:
    timestamp: float
    tool_name: str
    blocked: bool
    reason: str
    latency_ms: float


class ToolSandbox:
    def __init__(self) -> None:
        self._records: List[ToolExecutionRecord] = []
        self._lock = Lock()

    def execute(self, tool_calls: List[Dict]) -> Dict:
        results = []
        for i, tc in enumerate(tool_calls):
            name = str(tc.get("name", "unknown"))
            args = tc.get("arguments", {}) or {}
            blocked = any(x in str(args).lower() for x in ["drop table", "../", "rm -rf", "shutdown"])
            reason = "blocked_dangerous_pattern" if blocked else "allowed"
            latency_ms = 12.0 + i
            rec = ToolExecutionRecord(
                timestamp=time(),
                tool_name=name,
                blocked=blocked,
                reason=reason,
                latency_ms=latency_ms,
            )
            with self._lock:
                self._records.append(rec)
            results.append(
                {
                    "tool_name": name,
                    "arguments": args,
                    "blocked": blocked,
                    "block_reason": reason if blocked else "",
                    "result": {"ok": not blocked},
                    "latency_ms": latency_ms,
                }
            )
        return {"results": results, "stats": self.stats()}

    def stats(self) -> Dict:
        with self._lock:
            total = len(self._records)
            blocked = sum(1 for r in self._records if r.blocked)
            return {"total_calls": total, "blocked_calls": blocked, "block_rate": (blocked / total) if total else 0.0}

    def log(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [asdict(r) for r in self._records[-max(1, limit) :]]


class SessionSandbox:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, Dict] = {}

    def create(self, metadata: Optional[Dict] = None) -> Dict:
        sid = f"sess-{int(time() * 1000)}"
        payload = {
            "session_id": sid,
            "created_at": time(),
            "status": "active",
            "metadata": metadata or {},
            "messages": [],
        }
        with self._lock:
            self._sessions[sid] = payload
        return payload

    def append(self, session_id: str, role: str, content: str) -> Dict:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return {"error": "session_not_found", "session_id": session_id}
            sess["messages"].append({"ts": time(), "role": role, "content": content})
            return sess

    def get(self, session_id: str) -> Dict:
        with self._lock:
            return self._sessions.get(session_id, {"error": "session_not_found", "session_id": session_id})

    def close(self, session_id: str) -> Dict:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return {"error": "session_not_found", "session_id": session_id}
            sess["status"] = "closed"
            sess["closed_at"] = time()
            return sess


_TOOL_SANDBOX = ToolSandbox()
_SESSION_SANDBOX = SessionSandbox()


def get_tool_sandbox() -> ToolSandbox:
    return _TOOL_SANDBOX


def get_session_sandbox() -> SessionSandbox:
    return _SESSION_SANDBOX
