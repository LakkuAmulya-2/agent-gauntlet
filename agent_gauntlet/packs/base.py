from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PackResult:
    subscores: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class GauntletPack:
    """
    Standardized interface for production mistake packs.
    """

    name: str = "base"

    def inject(self, state: Dict[str, Any], step_ctx: Dict[str, Any]) -> Dict[str, Any]:
        return step_ctx

    def verify(self, action: Any, obs: Any, gt: Dict[str, Any]) -> PackResult:
        return PackResult()

    def metrics(self, episode_trace: List[Dict[str, Any]]) -> Dict[str, float]:
        return {}
