from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from threading import Lock
from time import time
from typing import Dict, List


@dataclass
class ComplianceAuditRecord:
    timestamp: float
    episode_id: str
    response_hash: str
    framework_violations: Dict[str, int]
    severity: str
    remediation: str


class ComplianceEngine:
    FRAMEWORKS = ["GDPR", "CCPA", "HIPAA", "SOC2", "EU_AI_ACT", "PCI_DSS"]

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: List[ComplianceAuditRecord] = []

    def add_record(self, episode_id: str, response_text: str, violation_count: int) -> None:
        severity = "LOW"
        if violation_count >= 3:
            severity = "HIGH"
        elif violation_count >= 1:
            severity = "MEDIUM"
        violations = {k: 0 for k in self.FRAMEWORKS}
        if violation_count > 0:
            violations["SOC2"] = 1
            violations["EU_AI_ACT"] = 1
        remediation = "Review refusal strategy and tighten policy checks." if violation_count else "No action required."
        record = ComplianceAuditRecord(
            timestamp=time(),
            episode_id=episode_id,
            response_hash=sha256(response_text.encode("utf-8")).hexdigest(),
            framework_violations=violations,
            severity=severity,
            remediation=remediation,
        )
        with self._lock:
            self._records.append(record)

    def report(self) -> Dict:
        with self._lock:
            total = len(self._records)
            violating = sum(1 for r in self._records if sum(r.framework_violations.values()) > 0)
            severities = {
                "LOW": sum(1 for r in self._records if r.severity == "LOW"),
                "MEDIUM": sum(1 for r in self._records if r.severity == "MEDIUM"),
                "HIGH": sum(1 for r in self._records if r.severity == "HIGH"),
            }
            return {
                "frameworks_covered": list(self.FRAMEWORKS),
                "total_records": total,
                "violation_rate": round((violating / total), 4) if total else 0.0,
                "severity_breakdown": severities,
            }

    def recent_records(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            return [asdict(r) for r in self._records[-max(1, limit) :]]


_GLOBAL_ENGINE = ComplianceEngine()


def get_compliance_engine() -> ComplianceEngine:
    return _GLOBAL_ENGINE
