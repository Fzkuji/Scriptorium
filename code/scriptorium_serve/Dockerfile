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

# The writer talks to an OpenAI-compatible endpoint over HTTP, so no Node
# runtime and no Claude Code CLI are needed. The workspace shell tool does use
# ordinary POSIX utilities, which the base image already provides.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY code/scriptorium_serve/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

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

# Add holds a request open for the whole write, so a single worker serves the
# platform's fan-out from the thread pool rather than from extra processes. The
# backlog covers the burst that arrives when 64 Add and 256 Search workers all
# connect at once.
CMD ["uvicorn", "scriptorium_serve.server:app", \
     "--app-dir", "/app/code", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "1200", \
     "--backlog", "512"]
