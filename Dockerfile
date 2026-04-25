FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

# Copy source
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GAUNTLET_ENV_URL=http://localhost:7860

CMD ["python", "demo_app.py"]
