"""FastAPI application for ScentMatch v0.6."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from scentmatch import __version__
from scentmatch.service import RecommendationService


class RecommendRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class ReasonResponse(BaseModel):
    type: str
    text: str
    strength: str
    score: float


class DrawbackResponse(BaseModel):
    type: str
    text: str
    agreement: float
    votes: int
    source_language: str


class ExplanationResponse(BaseModel):
    short_explanation: str
    why_recommended: list[ReasonResponse]
    possible_drawbacks: list[DrawbackResponse]
    evidence_level: str


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    perfume_id: str
    name: str
    brand: str
    score: float
    components: dict[str, float | None]
    weights: dict[str, float]
    accords: list[str]
    notes: dict[str, list[str]]
    gender: str | None = None
    year: int | None = None
    source_url: str | None = None
    community_evidence: dict[str, list[dict[str, Any]]]
    explanation: ExplanationResponse


class RecommendResponse(BaseModel):
    schema_version: str
    query: str
    query_summary: list[str]
    intent: dict[str, Any]
    reference: dict[str, Any] | None
    ranking_mode: str
    link_predictor_enabled: bool
    warnings: list[str]
    gate_counts: dict[str, int]
    results: list[RecommendationResponse]
    latency_ms: float


def create_app(
    service: RecommendationService | None = None,
    *,
    ui_path: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.recommendation_service = service
        else:
            app.state.recommendation_service = await run_in_threadpool(
                RecommendationService
            )
        yield

    app = FastAPI(
        title="ScentMatch API",
        description=(
            "Evidence-backed perfume recommendations using fine-tuned E5, "
            "community signals and leakage-safe link prediction."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    def get_service() -> RecommendationService:
        loaded = getattr(app.state, "recommendation_service", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        return loaded

    service_dependency = Depends(get_service)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url=ui_path or "/docs")

    @app.get("/health", tags=["system"])
    def health(
        loaded: RecommendationService = service_dependency,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "catalog_size": len(loaded.engine.profiles_by_id),
        }

    @app.get("/api/v1/info", tags=["system"])
    def info(
        loaded: RecommendationService = service_dependency,
    ) -> dict[str, Any]:
        return loaded.info()

    @app.post(
        "/api/v1/recommend",
        response_model=RecommendResponse,
        tags=["recommendations"],
    )
    async def recommend(
        request: RecommendRequest,
        loaded: RecommendationService = service_dependency,
    ) -> dict[str, Any]:
        try:
            return await run_in_threadpool(
                loaded.recommend, request.query, request.top_k
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/perfumes/{perfume_id}", tags=["catalogue"])
    def perfume(
        perfume_id: str,
        loaded: RecommendationService = service_dependency,
    ) -> dict[str, Any]:
        result = loaded.get_perfume(perfume_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Perfume not found")
        return result

    return app


def api_main() -> None:
    parser = argparse.ArgumentParser(description="Run the ScentMatch FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError('App dependencies are missing. Install ".[app]".') from error
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    api_main()
