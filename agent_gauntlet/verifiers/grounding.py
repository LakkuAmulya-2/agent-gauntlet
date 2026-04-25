from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GroundingDecision:
    grounded: bool
    violations: List[str]
    score: float
    evidence_links: int


class GroundingVerifier:
    """
    Formal-ish grounding verifier using provenance links + tenant boundary checks.
    """

    def verify(
        self,
        reasoning: str,
        ledger: Dict[str, Any],
        response: Dict[str, Any] | None,
        expected_tenant: str | None,
        observed_tenant: str | None,
    ) -> GroundingDecision:
        text = (reasoning or "").lower()
        facts = ledger.get("retrieved_facts", [])
        tool_outputs = ledger.get("tool_outputs", [])
        violations: List[str] = []

        evidence_links = 0
        for idx, fact in enumerate(facts):
            token = str(fact.get("fact_id", f"fact_{idx}")).lower()
            if token in text:
                evidence_links += 1

        if "definitely" in text and evidence_links == 0:
            violations.append("unsupported_claim")

        if response and response.get("_semantic_drift"):
            violations.append("semantic_contradiction")

        if response and response.get("_injected_payload") and "validated" in text:
            violations.append("fabricated_tool_output")

        if expected_tenant and observed_tenant and expected_tenant != observed_tenant:
            violations.append("wrong_tenant")

        # provenance completeness
        if tool_outputs and evidence_links == 0 and "because" in text:
            violations.append("missing_provenance_link")

        score = 1.0
        for v in violations:
            if v in {"unsupported_claim", "missing_provenance_link"}:
                score -= 0.25
            else:
                score -= 0.35
        score = max(0.0, min(1.0, score))

        return GroundingDecision(
            grounded=score >= 0.75 and not violations,
            violations=violations,
            score=score,
            evidence_links=evidence_links,
        )
