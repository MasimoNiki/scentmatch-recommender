"""Reusable, single-load retrieval engine for batch evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from scentmatch.data.audit import read_jsonl
from scentmatch.retrieval.community_ranker import (
    DEFAULT_COMMUNITY_PROFILES,
    CommunityFeatureStore,
    percentile_features,
    query_dependent_weights,
    weighted_score,
)
from scentmatch.retrieval.community_text import CommunityTextIndex
from scentmatch.retrieval.constraints import violates_exclusions
from scentmatch.retrieval.e5 import (
    DEFAULT_COMMUNITY_SIGNALS,
    DEFAULT_LINK_MODEL,
    DEFAULT_PRICE_SNAPSHOT,
    DEFAULT_PROFILES,
    DEFAULT_SEARCH_ARTIFACT_DIR,
    _diversify,
    _load_artifacts,
    _structured_fit,
    load_model,
)
from scentmatch.retrieval.intent import gender_matches_intent, parse_query_intent
from scentmatch.retrieval.reference import (
    display_name,
    gender_compatible,
    load_optional_index,
)
from scentmatch.training.link_prediction import CommunityLinkPredictor

RankingMode = Literal["semantic", "community", "perception"]


class HybridSearchEngine:
    """Load the model once and expose comparable ranking variants."""

    def __init__(
        self,
        profiles_path: Path = DEFAULT_PROFILES,
        artifact_dir: Path = DEFAULT_SEARCH_ARTIFACT_DIR,
        community_profiles_path: Path = DEFAULT_COMMUNITY_PROFILES,
        perception_artifact_dir: Path | None = None,
        community_signals_path: Path = DEFAULT_COMMUNITY_SIGNALS,
        prices_path: Path = DEFAULT_PRICE_SNAPSHOT,
        link_model_path: Path = DEFAULT_LINK_MODEL,
        enable_link_model: bool = True,
        device: str = "cpu",
    ) -> None:
        self.profiles_path = profiles_path.resolve()
        self.profiles = read_jsonl(self.profiles_path)
        self.profiles_by_id = {
            str(profile["perfume_id"]): profile for profile in self.profiles
        }
        self.embeddings, self.metadata = _load_artifacts(
            artifact_dir.resolve(), self.profiles_path
        )
        self.model = load_model(self.metadata["model"], device)
        self.perfume_ids = [str(value) for value in self.metadata["perfume_ids"]]
        self.id_to_index = {
            perfume_id: index for index, perfume_id in enumerate(self.perfume_ids)
        }
        self.community_store = (
            CommunityFeatureStore.load(community_profiles_path.resolve())
            if community_profiles_path.exists()
            else None
        )
        self.perception_index = (
            CommunityTextIndex.load(perception_artifact_dir.resolve())
            if perception_artifact_dir is not None
            and (perception_artifact_dir / "metadata.json").exists()
            else None
        )
        self.legacy_community = load_optional_index(community_signals_path.resolve())
        self.prices = load_optional_index(prices_path.resolve())
        resolved_link_model = link_model_path.resolve()
        self.link_predictor = (
            CommunityLinkPredictor.load(
                resolved_link_model,
                self.profiles,
                list(self.community_store.rows.values()),
                self.embeddings,
                self.perfume_ids,
            )
            if enable_link_model
            and self.community_store is not None
            and resolved_link_model.exists()
            else None
        )

    def encode_queries(
        self, query_texts: list[str], batch_size: int = 16
    ) -> np.ndarray:
        return self.model.encode(
            [f"query: {query}" for query in query_texts],
            batch_size=batch_size,
            show_progress_bar=len(query_texts) > 1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def rank(
        self,
        query: str,
        mode: RankingMode = "community",
        top_k: int = 10,
        query_embedding: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if mode != "semantic" and self.community_store is None:
            raise FileNotFoundError("Community profiles are required for this mode")
        if mode == "perception" and self.perception_index is None:
            raise FileNotFoundError(
                "Build community text artifacts before perception evaluation"
            )
        if query_embedding is None:
            query_embedding = self.encode_queries([query])[0]
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)

        intent = parse_query_intent(query, self.profiles)
        resolution = intent.reference
        reference = (
            resolution.profile
            if resolution is not None and resolution.accepted
            else None
        )
        reference_id = str(reference["perfume_id"]) if reference is not None else None
        semantic_raw = {
            perfume_id: float(score)
            for perfume_id, score in zip(
                self.perfume_ids,
                (query_embedding @ self.embeddings.T)[0],
                strict=True,
            )
        }
        semantic_percentiles = percentile_features(semantic_raw)
        reference_percentiles: dict[str, float] = {}
        if reference_id is not None:
            reference_vector = self.embeddings[
                self.id_to_index[reference_id] : self.id_to_index[reference_id] + 1
            ]
            reference_percentiles = percentile_features(
                {
                    perfume_id: float(score)
                    for perfume_id, score in zip(
                        self.perfume_ids,
                        (reference_vector @ self.embeddings.T)[0],
                        strict=True,
                    )
                }
            )

        perception_matches = (
            self.perception_index.score(query_embedding[0])
            if mode == "perception" and self.perception_index is not None
            else {}
        )
        exclusions = {
            "notes": set(intent.excluded_notes),
            "accords": set(intent.excluded_accords),
        }
        preferences = set(intent.preferred_notes) | set(intent.preferred_accords)
        weights = query_dependent_weights(intent, reference is not None)
        reference_price = self.prices.get(reference_id or "", {}).get("median_price")
        budget_intent = intent.budget_mode != "any"
        candidates: list[dict[str, Any]] = []
        gate_counts: dict[str, int] = {}

        for perfume_id in self.perfume_ids:
            if perfume_id == reference_id:
                continue
            profile = self.profiles_by_id[perfume_id]
            if reference is not None and not gender_compatible(reference, profile):
                continue
            if not gender_matches_intent(intent, profile):
                continue
            if violates_exclusions(profile, exclusions):
                continue

            components: dict[str, float | None] = {
                "semantic_query": semantic_percentiles[perfume_id]
            }
            used_weights: dict[str, float] = {"semantic_query": 1.0}
            score = semantic_percentiles[perfume_id]
            if mode != "semantic" and self.community_store is not None:
                gate = self.community_store.violates_community_gate(perfume_id, intent)
                if gate is not None:
                    gate_counts[gate] = gate_counts.get(gate, 0) + 1
                    continue
                components.update(
                    self.community_store.features(
                        perfume_id, intent, reference_id
                    ).to_dict()
                )
                if self.link_predictor is not None and reference_id is not None:
                    components["link_prediction"] = self.link_predictor.score(
                        reference_id, perfume_id
                    )
                components.update(
                    {
                        "reference_content": reference_percentiles.get(perfume_id),
                        "structured_fit": _structured_fit(profile, preferences),
                    }
                )
                score, used_weights = weighted_score(components, weights)

            match = perception_matches.get(perfume_id)
            if match is not None:
                components["community_text_similarity"] = match.score
                score = 0.88 * score + 0.12 * match.score

            if budget_intent and reference_price is not None:
                candidate_price = self.prices.get(perfume_id, {}).get("median_price")
                if candidate_price is None or candidate_price >= reference_price:
                    continue
                budget_fit = 1.0 - candidate_price / reference_price
                components["budget_fit"] = budget_fit
                score += 0.02 * budget_fit
            elif budget_intent:
                if (
                    self.legacy_community.get(perfume_id, {})
                    .get("segment", "")
                    .casefold()
                    != "mass-market"
                ):
                    continue
                components["budget_fit"] = 1.0

            candidates.append(
                {
                    "perfume_id": perfume_id,
                    "profile": profile,
                    "score": score,
                    "semantic_raw": semantic_raw[perfume_id],
                    "components": components,
                    "weights": used_weights,
                    "perception_evidence": (
                        match.statement if match is not None else None
                    ),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = _diversify(
            candidates,
            self.embeddings,
            self.id_to_index,
            top_k,
            reference is not None,
        )
        results = []
        for rank, item in enumerate(kept, start=1):
            perfume_id = item["perfume_id"]
            consensus = (
                self.community_store.consensus(perfume_id)
                if self.community_store is not None
                else {}
            )
            results.append(
                {
                    "rank": rank,
                    "perfume_id": perfume_id,
                    "name": display_name(str(item["profile"]["name"])),
                    "brand": display_name(str(item["profile"]["brand"])),
                    "score": round(float(item["final_score"]), 6),
                    "base_score": round(float(item["score"]), 6),
                    "semantic_raw": round(float(item["semantic_raw"]), 6),
                    "weights": {
                        key: round(float(value), 6)
                        for key, value in item["weights"].items()
                    },
                    "diversity_adjustment": round(
                        float(item["diversity_adjustment"]), 6
                    ),
                    "components": {
                        key: round(float(value), 6) if value is not None else None
                        for key, value in item["components"].items()
                    },
                    "accords": item["profile"].get("accords", [])[:8],
                    "notes": item["profile"].get("notes", {}),
                    "gender": item["profile"].get("gender"),
                    "year": item["profile"].get("year"),
                    "source_url": item["profile"].get("source_url"),
                    "community_evidence": {
                        "pros": consensus.get("pros", [])[:2],
                        "cons": consensus.get("cons", [])[:2],
                    },
                    "perception_evidence": item["perception_evidence"],
                }
            )
        return {
            "schema_version": "6.0",
            "query": query,
            "mode": mode,
            "intent": intent.to_dict(),
            "reference_id": reference_id,
            "link_predictor_enabled": self.link_predictor is not None,
            "active_weights": weights,
            "gate_counts": gate_counts,
            "results": results,
        }
