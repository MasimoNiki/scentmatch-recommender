# Application, runtime assets and deployment

This document consolidates the public application contract, runtime-asset policy, Docker packaging and CI behaviour.

## Runtime flow

```text
query
  -> QueryIntent + reference resolution
  -> fine-tuned E5 and structured/community features
  -> optional link-prediction score for reference queries
  -> constraints and diversity
  -> ExplanationGenerator
  -> CLI / REST API / Gradio
```

`RecommendationService` is the shared inference entry point. The model, document embeddings, community feature store and link predictor are loaded once when the service starts, so CLI, API and UI use the same ranking code.

## REST API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | readiness and catalogue size |
| `GET` | `/api/v1/info` | model and service metadata |
| `POST` | `/api/v1/recommend` | recommendations with explanations |
| `GET` | `/api/v1/perfumes/{id}` | catalogue and community profile |

Example request:

```json
{
  "query": "Стойкий свежий аромат на лето без сильного шлейфа",
  "top_k": 5
}
```

Input is limited to 500 characters and `top_k` to 1–10. CPU-bound inference runs outside the async event loop.

## Explanation contract

Each recommendation may contain:

- typed `why_recommended` reasons;
- vote-backed `possible_drawbacks`;
- `evidence_level` (`high`, `medium`, `limited`);
- raw scoring components and active weights.

A reason is emitted only when its source signal exists. The evidence level describes the amount of supporting evidence, not the probability that a user will like the fragrance.

## Runtime assets

Large and derived inference artifacts intentionally stay outside normal Git history. Run:

```powershell
scentmatch-runtime-check
```

A complete local demo expects paths such as:

```text
data/processed/perfume_profiles_5000.jsonl
data/augmentation/community_profiles_v2.jsonl
data/augmentation/community_signals.jsonl
data/augmentation/price_snapshot_2024.jsonl
artifacts/e5_finetuned/document_embeddings.npy
artifacts/e5_finetuned/metadata.json
artifacts/link_predictor_v1/model.json
models/e5_finetuned/config.json
models/e5_finetuned/modules.json
models/e5_finetuned/*.safetensors
```

Small evidence artifacts are allowed in Git (model metadata, portable link predictor, benchmark reports and evaluation sets). Model weights, document embeddings and the derived 5,000-item catalogue are excluded.

The current public portfolio uses screenshots and preserved benchmark/evaluation reports instead of redistributing source-derived runtime data or large model weights.

## Docker

Docker is a packaging/deployment demonstration rather than the primary development environment.

```powershell
scentmatch-runtime-check
docker compose up --build
```

The local image installs CPU-only PyTorch and does not require CUDA/NVIDIA packages. The application runs as a non-root user and exposes an HTTP health check. Local Compose requires runtime assets and fails early when they are missing.

After the check:

```powershell
docker compose down
```

The image remains on disk until explicitly removed, for example:

```powershell
docker image rm scentmatch:0.7.0
```

## CI

GitHub Actions runs the lightweight public checks on every push to `main` and pull request:

1. install the project on Python 3.11 and 3.12;
2. run Ruff;
3. run pytest;
4. verify that a code-only Docker image can be built without model weights or large runtime assets.

Model-dependent online inference is deliberately not executed in CI; API/service/UI contracts use deterministic test doubles where appropriate.

## Docker/PyPI TLS troubleshooting

If Docker fails while accessing `https://pypi.org/simple/...` with an SSL/TLS error, first verify container connectivity:

```powershell
docker run --rm python:3.11-slim-bookworm python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org/simple/setuptools/', timeout=20).status)"
```

Expected output: `200`.

If DNS succeeds but TLS fails, check **Docker Desktop -> Settings -> Resources -> Proxies**, VPN/firewall settings and HTTPS-inspection certificates. Do not work around certificate problems with pip `--trusted-host` and never commit proxy credentials.

After fixing connectivity:

```powershell
docker compose build --no-cache --progress=plain
docker compose up
```

This class of error happens before the ScentMatch package is installed and is therefore a Docker/network configuration problem rather than a ranking-model failure.

## Current limitations

- similarity and community perception remain subjective;
- free-form ranking does not yet have a large expert-judged test set;
- the link predictor is used only for resolved-reference queries;
- price data is historical rather than live availability;
- source licences restrict the current build to non-commercial research/portfolio use.
