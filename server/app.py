from __future__ import annotations

import os
import time

import gradio as gr
from openenv.core.env_server import create_fastapi_app

from agent_gauntlet.models import AgentAction, DifficultyLevel, TaskObservation
from agent_gauntlet.runtime.environment import AgentGauntletEnvironment

_difficulty_str = os.environ.get("GAUNTLET_DIFFICULTY", "easy").lower()
_difficulty_map = {
    "easy": DifficultyLevel.EASY,
    "medium": DifficultyLevel.MEDIUM,
    "hard": DifficultyLevel.HARD,
    "expert": DifficultyLevel.EXPERT,
}
_difficulty = _difficulty_map.get(_difficulty_str, DifficultyLevel.EASY)

_seed_str = os.environ.get("GAUNTLET_SEED", "")
_seed = int(_seed_str) if _seed_str.isdigit() else None
_start_time = time.time()


def _env_factory() -> AgentGauntletEnvironment:
    return AgentGauntletEnvironment(
        default_difficulty=_difficulty,
        seed=_seed,
    )


app = create_fastapi_app(_env_factory, AgentAction, TaskObservation)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "uptime_s": round(time.time() - _start_time, 1),
        "difficulty": _difficulty.value,
    }


@app.get("/kaizen/report")
def kaizen_report():
    """Kaizen self-improvement report — skill profile, learning curve, trace memory."""
    env = _env_factory()
    return env.kaizen_report


@app.get("/kaizen/sft_dataset")
def kaizen_sft_dataset():
    """Export trace memory as SFT training examples for next round."""
    env = _env_factory()
    return {"examples": env.export_sft_dataset()}


@app.get("/adversarial/stats")
def adversarial_stats():
    """Adversarial generator stats — top breaking combos, solver failure rate."""
    from agent_gauntlet.runtime.adversarial import get_global_generator
    return get_global_generator().stats()


@app.get("/counterfactual/stats")
def counterfactual_stats():
    """Counterfactual replay stats — avg regret, high-regret steps."""
    from agent_gauntlet.runtime.counterfactual import get_global_engine
    return get_global_engine().stats()


@app.get("/critic/report")
def critic_report():
    """Live reward hacking detector report — patterns caught, penalties applied."""
    env = _env_factory()
    return env.critic_report


@app.post("/pareto/score")
def pareto_score(metadata: dict):
    """Compute Pareto scores (capability, safety, speed) for episode metadata."""
    env = _env_factory()
    return env.pareto_scores(metadata)


_ENABLE_WEB = os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() != "false"
if _ENABLE_WEB:
    try:
        os.environ.setdefault("GAUNTLET_ENV_URL", "http://localhost:8000")
        from demo_app import build_ui

        _gradio_app = build_ui()
        app = gr.mount_gradio_app(app, _gradio_app, path="/web")
    except Exception as _e:
        import warnings

        warnings.warn(f"Gradio web UI not available: {_e}")


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
