from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Any, Deque, Dict, List


@dataclass
class RuntimeMetricsState:
    episodes_started: int = 0
    steps_total: int = 0
    steps_done: int = 0
    violations_total: int = 0
    last_episode_id: str = ""
    last_reward: float = 0.0
    recent_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    recent_events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    trust_snapshots: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))


class RuntimeMetricsStore:
    """Shared live runtime metrics fed from actual /reset and /step traffic."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = RuntimeMetricsState()

    def record_reset(self, payload: Dict[str, Any]) -> None:
        obs = payload.get("observation", {}) if isinstance(payload, dict) else {}
        with self._lock:
            self._state.episodes_started += 1
            self._state.last_episode_id = str(obs.get("episode_id", ""))
            self._state.recent_events.append(
                {
                    "ts": time(),
                    "kind": "reset",
                    "episode_id": self._state.last_episode_id,
                    "difficulty": obs.get("difficulty", ""),
                }
            )

    def record_step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        obs = payload.get("observation", {}) if isinstance(payload, dict) else {}
        reward = float(payload.get("reward", 0.0)) if isinstance(payload, dict) else 0.0
        evidence = obs.get("verifier_evidence", []) or []
        violations = sum(len(ev.get("violations", [])) for ev in evidence if isinstance(ev, dict))
        with self._lock:
            self._state.steps_total += 1
            if bool(obs.get("is_done", False)):
                self._state.steps_done += 1
            self._state.violations_total += int(violations)
            self._state.last_reward = reward
            self._state.recent_rewards.append(reward)
            self._state.last_episode_id = str(obs.get("episode_id", self._state.last_episode_id))
            self._state.recent_events.append(
                {
                    "ts": time(),
                    "kind": "step",
                    "episode_id": self._state.last_episode_id,
                    "reward": reward,
                    "violations": int(violations),
                    "done": bool(obs.get("is_done", False)),
                }
            )
            return {
                "episode_id": self._state.last_episode_id,
                "violations": int(violations),
                "reward": reward,
            }

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            rewards = list(self._state.recent_rewards)
            avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
            return {
                "episodes_started": self._state.episodes_started,
                "steps_total": self._state.steps_total,
                "steps_done": self._state.steps_done,
                "violations_total": self._state.violations_total,
                "last_episode_id": self._state.last_episode_id,
                "last_reward": self._state.last_reward,
                "avg_reward_recent": avg_reward,
            }

    def events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._state.recent_events)[-max(1, limit) :]


_STORE = RuntimeMetricsStore()


def get_runtime_metrics_store() -> RuntimeMetricsStore:
    return _STORE
