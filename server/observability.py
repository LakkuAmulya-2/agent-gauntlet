from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from statistics import mean
from threading import Lock
from time import time
from typing import Deque, Dict, Iterable, List


@dataclass
class DecisionTrace:
    timestamp: float
    episode_id: str
    difficulty: str
    domain: str
    step: int
    reward: float
    violation_count: int
    termination_reason: str


class ObservabilityHub:
    """Thread-safe in-memory observability collector for live dashboards."""

    def __init__(self, max_events: int = 500) -> None:
        self._lock = Lock()
        self._events: Deque[Dict] = deque(maxlen=max_events)
        self._traces: Deque[DecisionTrace] = deque(maxlen=max_events)
        self._latencies_ms: Deque[float] = deque(maxlen=max_events)
        self._errors: Deque[str] = deque(maxlen=max_events)
        self._snapshots: Deque[Dict] = deque(maxlen=max_events)

    def record_event(self, kind: str, payload: Dict) -> None:
        with self._lock:
            self._events.append({"ts": time(), "kind": kind, "payload": payload})

    def record_trace(self, trace: DecisionTrace) -> None:
        with self._lock:
            self._traces.append(trace)

    def record_latency_ms(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(float(latency_ms))

    def record_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)

    def record_snapshot(self, snapshot: Dict) -> None:
        with self._lock:
            self._snapshots.append({"ts": time(), **snapshot})

    def recent_traces(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            return [asdict(t) for t in list(self._traces)[-max(1, limit) :]]

    def events(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            return list(self._events)[-max(1, limit) :]

    def full_report(self) -> Dict:
        with self._lock:
            lat = list(self._latencies_ms)
            p50 = _pct(lat, 50)
            p95 = _pct(lat, 95)
            p99 = _pct(lat, 99)
            return {
                "latency_ms": {
                    "count": len(lat),
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "avg": round(mean(lat), 2) if lat else 0.0,
                },
                "decision_traces": len(self._traces),
                "recent_errors": list(self._errors)[-10:],
                "recent_events": list(self._events)[-20:],
                "recent_snapshots": list(self._snapshots)[-20:],
            }

    def latest_snapshot(self) -> Dict:
        with self._lock:
            return dict(self._snapshots[-1]) if self._snapshots else {}

    def snapshot_history(self, limit: int = 60) -> List[Dict]:
        with self._lock:
            return list(self._snapshots)[-max(1, limit) :]

    def sse_lines(self, limit: int = 20) -> Iterable[str]:
        for ev in self.events(limit=limit):
            yield f"event: {ev['kind']}\n"
            yield f"data: {ev}\n\n"


def _pct(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * (percentile / 100.0))
    return round(ordered[idx], 2)


_GLOBAL_HUB = ObservabilityHub()


def get_observability_hub() -> ObservabilityHub:
    return _GLOBAL_HUB
