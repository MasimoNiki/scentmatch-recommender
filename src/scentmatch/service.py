"""Application-facing ScentMatch recommendation service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from scentmatch import __version__
from scentmatch.retrieval.community_ranker import DEFAULT_COMMUNITY_PROFILES
from scentmatch.retrieval.e5 import (
    DEFAULT_COMMUNITY_SIGNALS,
    DEFAULT_LINK_MODEL,
    DEFAULT_PRICE_SNAPSHOT,
    DEFAULT_PROFILES,
    DEFAULT_SEARCH_ARTIFACT_DIR,
)
from scentmatch.retrieval.engine import HybridSearchEngine
from scentmatch.retrieval.explanations import ExplanationGenerator, summarize_intent
from scentmatch.retrieval.reference import display_name


@dataclass(frozen=True)
class ServiceSettings:
    profiles_path: Path = DEFAULT_PROFILES
    artifact_dir: Path = DEFAULT_SEARCH_ARTIFACT_DIR
    community_profiles_path: Path = DEFAULT_COMMUNITY_PROFILES
    community_signals_path: Path = DEFAULT_COMMUNITY_SIGNALS
    prices_path: Path = DEFAULT_PRICE_SNAPSHOT
    link_model_path: Path = DEFAULT_LINK_MODEL
    device: str = "cpu"
    enable_link_model: bool = True


class RecommendationService:
    """One loaded engine shared by CLI, API and demo UI."""

    def __init__(
        self,
        settings: ServiceSettings | None = None,
        *,
        engine: HybridSearchEngine | Any | None = None,
    ) -> None:
        self.settings = settings or ServiceSettings()
        self.engine = engine or HybridSearchEngine(
            profiles_path=self.settings.profiles_path,
            artifact_dir=self.settings.artifact_dir,
            community_profiles_path=self.settings.community_profiles_path,
            community_signals_path=self.settings.community_signals_path,
            prices_path=self.settings.prices_path,
            link_model_path=self.settings.link_model_path,
            enable_link_model=self.settings.enable_link_model,
            device=self.settings.device,
        )
        self.explanations = ExplanationGenerator()

    def recommend(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        if len(query) > 500:
            raise ValueError("Query must be 500 characters or fewer")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        started = perf_counter()
        ranked = self.engine.rank(query, mode="community", top_k=top_k)
        intent = ranked["intent"]
        reference_id = ranked.get("reference_id")
        reference = self._reference_payload(reference_id, intent)
        reference_name = reference.get("display_name") if reference else None

        results = []
        for result in ranked["results"]:
            enriched = dict(result)
            enriched["explanation"] = self.explanations.explain(
                intent,
                result,
                reference_name=reference_name,
            )
            results.append(enriched)

        elapsed_ms = (perf_counter() - started) * 1000.0
        return {
            "schema_version": "6.0",
            "query": query,
            "query_summary": summarize_intent(intent),
            "intent": intent,
            "reference": reference,
            "ranking_mode": "hybrid_community_v2_link_prediction",
            "link_predictor_enabled": bool(ranked.get("link_predictor_enabled")),
            "warnings": self._warnings(intent, reference_id, ranked),
            "gate_counts": ranked.get("gate_counts", {}),
            "results": results,
            "latency_ms": round(elapsed_ms, 2),
        }

    def get_perfume(self, perfume_id: str) -> dict[str, Any] | None:
        profile = self.engine.profiles_by_id.get(str(perfume_id))
        if profile is None:
            return None
        payload = dict(profile)
        payload["name"] = display_name(str(profile["name"]))
        payload["brand"] = display_name(str(profile["brand"]))
        community_store = getattr(self.engine, "community_store", None)
        if community_store is not None:
            community = community_store.rows.get(str(perfume_id), {})
            payload["community"] = {
                "rating": community.get("rating", {}),
                "longevity": community.get("longevity", {}),
                "sillage": community.get("sillage", {}),
                "seasons": community.get("seasons", {}),
                "daypart": community.get("daypart", {}),
                "summary": community.get("community_summary", {}),
            }
        return payload

    def info(self) -> dict[str, Any]:
        metadata = getattr(self.engine, "metadata", {})
        return {
            "name": "ScentMatch",
            "version": __version__,
            "catalog_size": len(self.engine.profiles_by_id),
            "embedding_model": metadata.get("model"),
            "embedding_dimension": metadata.get("embedding_dimension"),
            "link_predictor_enabled": getattr(
                self.engine, "link_predictor", None
            )
            is not None,
            "commercial_use": False,
        }

    def _reference_payload(
        self, reference_id: str | None, intent: dict[str, Any]
    ) -> dict[str, Any] | None:
        resolution = intent.get("reference")
        if reference_id is None:
            return resolution
        profile = self.engine.profiles_by_id[reference_id]
        payload = dict(resolution or {})
        payload.update(
            {
                "perfume_id": reference_id,
                "name": display_name(str(profile["name"])),
                "brand": display_name(str(profile["brand"])),
                "display_name": (
                    f"{display_name(str(profile['name']))} — "
                    f"{display_name(str(profile['brand']))}"
                ),
            }
        )
        return payload

    def _warnings(
        self,
        intent: dict[str, Any],
        reference_id: str | None,
        ranked: dict[str, Any],
    ) -> list[str]:
        language = "ru" if intent.get("language") == "ru" else "en"
        warnings: list[str] = []
        resolution = intent.get("reference")
        if resolution and not resolution.get("accepted", False):
            warnings.append(
                "Не удалось надёжно определить аромат-референс; система не стала "
                "угадывать."
                if language == "ru"
                else "The reference perfume could not be resolved reliably; the "
                "system did not guess."
            )
        if reference_id is not None and not ranked.get("link_predictor_enabled"):
            warnings.append(
                "ML-модель сходства недоступна; использованы остальные сигналы."
                if language == "ru"
                else "The ML similarity model is unavailable; other signals were used."
            )
        if intent.get("budget_mode") != "any":
            warnings.append(
                "Бюджетное ограничение использует неполный исторический snapshot "
                "или proxy по сегменту и не отражает текущие цены."
                if language == "ru"
                else "The budget constraint uses an incomplete historical snapshot "
                "or a segment proxy and does not represent current prices."
            )
        return warnings
