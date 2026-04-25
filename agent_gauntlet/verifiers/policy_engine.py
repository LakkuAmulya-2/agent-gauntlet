from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PolicyDecision:
    allowed: bool
    risk: str
    violations: List[str]
    reason: str


class PolicyEngine:
    """
    Enterprise-style policy engine with semantic tool-argument checks.
    """

    HIGH_RISK_PATTERNS = (
        r"drop\s+table",
        r"truncate\s+table",
        r"delete\s+all",
        r"disable[_\s-]*guardrail",
        r"exfiltrat",
        r"raw[_\s-]*phi",
        r"card[_\s-]*number",
    )

    def evaluate_action(
        self,
        action_type: str,
        tool_name: str | None,
        tool_args: Dict[str, Any] | None,
        policy: Dict[str, Any] | None,
        risk_hint: str | None = None,
    ) -> PolicyDecision:
        tool = tool_name or ""
        args = tool_args or {}
        pol = policy or {}
        forbidden = set(pol.get("forbidden_actions", []))
        violations: List[str] = []

        if tool and tool in forbidden:
            violations.append("forbidden_tool")

        arg_blob = " ".join(f"{k}={v}" for k, v in sorted(args.items())).lower()
        for pat in self.HIGH_RISK_PATTERNS:
            if re.search(pat, arg_blob):
                violations.append("unsafe_tool_argument")
                break

        risk = risk_hint or "low"
        if "unsafe_tool_argument" in violations or "forbidden_tool" in violations:
            risk = "high"
        elif risk == "low" and ("pii" in arg_blob or "tenant" in arg_blob):
            risk = "med"

        allowed = len(violations) == 0
        reason = "allowed" if allowed else f"blocked: {', '.join(violations)}"
        return PolicyDecision(allowed=allowed, risk=risk, violations=violations, reason=reason)
