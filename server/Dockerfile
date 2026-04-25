ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install environment dependencies (includes gradio)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy source
COPY src/openenv/core/ /app/src/openenv/core/
COPY agent_gauntlet/ /app/agent_gauntlet/
COPY server/ /app/server/
# Copy Gradio demo app
COPY demo_app.py /app/demo_app.py

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables with defaults
ENV GAUNTLET_DIFFICULTY=easy
ENV GAUNTLET_SEED=""
ENV WORKERS=4
ENV MAX_CONCURRENT_ENVS=100
ENV ENABLE_WEB_INTERFACE=true

# Run server (FastAPI + Gradio at /web) using OpenEnv template-style entrypoint
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
