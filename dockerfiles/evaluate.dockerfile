FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim



RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# RUN uv sync --frozen --no-install-project

COPY uv.lock uv.lock
COPY ml_ops/requirements.txt ml_ops/requirements.txt
COPY pyproject.toml pyproject.toml
COPY ml_ops/src/ src/
COPY ml_ops/data/ data/
COPY ml_ops/models models/

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync


ENTRYPOINT ["uv", "run", "src/ml_ops/evaluate.py"]

