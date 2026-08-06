# Add / Search service for the Agent Memory Leaderboard.
#
# Build from the repository root:
#   docker build -f code/scriptorium_serve/Dockerfile -t scriptorium-memory .
#
# Run (writer credentials are injected, never baked into the image):
#   docker run -p 8000:8000 \
#     -e SCRIPTORIUM_WRITER_BASE_URL=... \
#     -e SCRIPTORIUM_WRITER_API_KEY=... \
#     -e SCRIPTORIUM_SERVICE_TOKEN=... \
#     -v scriptorium-workspaces:/data/workspaces \
#     scriptorium-memory
FROM python:3.12-slim

# The writer runs Claude Code as a subprocess, so the CLI and a Node runtime
# have to be present in the image alongside the Python service.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"

COPY code/ ./code/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/code \
    SCRIPTORIUM_WORKSPACES=/data/workspaces \
    SCRIPTORIUM_WRITER_MODEL=gpt-4o-mini

RUN mkdir -p /data/workspaces
VOLUME ["/data/workspaces"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Add holds a request open for the whole write, so keep workers modest and let
# the platform's own concurrency drive throughput.
CMD ["uvicorn", "scriptorium_serve.server:app", \
     "--app-dir", "/app/code", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "1200"]
