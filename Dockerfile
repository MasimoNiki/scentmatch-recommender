# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG REQUIRE_RUNTIME_ASSETS=0
ARG INSTALL_INFERENCE=1
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV HF_HOME=/tmp/huggingface \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_RETRIES=10 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 scentmatch \
    && useradd --uid 10001 --gid 10001 --create-home scentmatch

COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "setuptools>=68" wheel \
    && if [ "$INSTALL_INFERENCE" = "1" ]; then \
        python -m pip install --index-url "$PYTORCH_INDEX_URL" "torch>=2.2" \
        && python -m pip install --no-build-isolation ".[inference,app]"; \
    else \
        python -m pip install --no-build-isolation ".[app]"; \
    fi

COPY . .

RUN if [ "$REQUIRE_RUNTIME_ASSETS" = "1" ]; then \
        scentmatch-runtime-check --root /app; \
    fi \
    && chown -R scentmatch:scentmatch /app

USER scentmatch

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

CMD ["scentmatch-app", "--host", "0.0.0.0", "--port", "8000"]
