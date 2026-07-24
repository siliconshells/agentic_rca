# Single image, two entrypoints (the API and the MCP server) selected by compose command.
# Multi-stage so the runtime image carries only the installed venv, not the uv build cache.
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY mcp_server ./mcp_server
COPY evals ./evals
# Install into a venv we can copy wholesale into the runtime stage.
RUN uv venv /app/.venv && VIRTUAL_ENV=/app/.venv uv pip install -e "."

FROM python:3.12-slim AS runtime

# Non-root: the harness runs untrusted-influenced content; nothing here needs root.
RUN useradd --create-home --uid 10001 aegis
WORKDIR /app

COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AEGIS_DB_PATH=/data/aegis.db

RUN mkdir -p /data && chown -R aegis:aegis /app /data
USER aegis

# Overridden by compose per service; the API is the sensible default.
EXPOSE 8000
CMD ["uvicorn", "aegis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
