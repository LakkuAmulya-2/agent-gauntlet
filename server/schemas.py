from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    difficulty: str


class TrustInputs(BaseModel):
    health_ok: float
    reward_component: float
    adversarial_component: float
    counterfactual_component: float
    critic_component: float


class TrustScoreResponse(BaseModel):
    score: float
    grade: str
    ready_for_production: bool
    inputs: TrustInputs


class KaizenSummary(BaseModel):
    enabled: bool
    episode_count: int
    avg_reward_recent: float
    learning_curve_points: int


class MetricsResponse(BaseModel):
    health: HealthResponse
    kaizen: KaizenSummary
    adversarial: Dict[str, Any]
    counterfactual: Dict[str, Any]
    critic: Dict[str, Any]
    trust: TrustScoreResponse


class CertificateResponse(BaseModel):
    certified: bool
    badge: str
    certificate_id: str
    score: float
    grade: str


class LiveMetricsResponse(BaseModel):
    history: List[Dict[str, Any]]
