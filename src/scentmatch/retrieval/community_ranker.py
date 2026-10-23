"""Community features and query-dependent ranking for ScentMatch v0.3."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl
from scentmatch.retrieval.intent import QueryIntent

DEFAULT_COMMUNITY_PROFILES = Path(
    "data/augmentation/community_profiles_v2.jsonl"
)


def _rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    denominator = max(1, len(ordered) - 1)
    return {key: rank / denominator for rank, key in enumerate(ordered)}


def percentile_features(values: dict[str, float]) -> dict[str, float]:
    """Public helper used to calibrate per-query cosine scores to 0..1."""
    return _rank_percentiles(values)


def _sum_votes(values: dict[str, Any]) -> int:
    return sum(max(0, int(value or 0)) for value in values.values())


def _edge_agreement(up_votes: int, down_votes: int) -> float:
    """Agreement above chance, shrunken for low-support comparisons."""
    up = max(0, int(up_votes))
    down = max(0, int(down_votes))
    votes = up + down
    posterior = (up + 2.0) / (votes + 4.0)
    agreement = max(0.0, 2.0 * (posterior - 0.5))
    reliability = 1.0 - math.exp(-votes / 30.0)
    return agreement * reliability


def _combine_evidence(first: float, second: float) -> float:
    return 1.0 - (1.0 - first) * (1.0 - second)


@dataclass(frozen=True)
class CommunityCandidateFeatures:
    season_fit: float | None
    daypart_fit: float | None
    longevity_fit: float | None
    sillage_fit: float | None
    sweetness_fit: float | None
    popularity: float
    quality: float
    community_gender_fit: float | None
    reminds_me_of: float | None
    also_liked: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "season_fit": self.season_fit,
            "daypart_fit": self.daypart_fit,
            "longevity_fit": self.longevity_fit,
            "sillage_fit": self.sillage_fit,
            "sweetness_fit": self.sweetness_fit,
            "popularity": self.popularity,
            "quality": self.quality,
            "community_gender_fit": self.community_gender_fit,
            "reminds_me_of": self.reminds_me_of,
            "also_liked": self.also_liked,
        }


class CommunityFeatureStore:
    """Small in-memory index over the selected 5k community profiles."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {str(row["perfume_id"]): row for row in rows}
        self._reverse_reminds: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._reverse_likes: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for source_id, row in self.rows.items():
            for edge in row.get("similarity", {}).get("reminds_me_of", []):
                score = _edge_agreement(edge["up_votes"], edge["down_votes"])
                self._reverse_reminds[str(edge["perfume_id"])].append(
                    (source_id, score)
                )
            for rank, target_id in enumerate(
                row.get("similarity", {}).get("also_liked", []), start=1
            ):
                self._reverse_likes[str(target_id)].append(
                    (source_id, math.exp(-(rank - 1) / 8.0))
                )

        season_totals: dict[str, int] = defaultdict(int)
        daypart_totals: dict[str, int] = defaultdict(int)
        for row in rows:
            for key, value in row.get("seasons", {}).items():
                season_totals[key] += max(0, int(value or 0))
            for key, value in row.get("daypart", {}).items():
                daypart_totals[key] += max(0, int(value or 0))
        season_sum = max(1, sum(season_totals.values()))
        daypart_sum = max(1, sum(daypart_totals.values()))
        self.season_priors = {
            key: value / season_sum for key, value in season_totals.items()
        }
        self.daypart_priors = {
            key: value / daypart_sum for key, value in daypart_totals.items()
        }

        popularity_raw = {}
        quality_raw = {}
        weighted_rating_sum = 0.0
        rating_votes_sum = 0
        for perfume_id, row in self.rows.items():
            popularity = row.get("popularity", {})
            popularity_raw[perfume_id] = (
                0.50 * math.log1p(max(0, int(popularity.get("magnitude") or 0)))
                + 0.30
                * math.log1p(max(0, int(popularity.get("compound_magnitude") or 0)))
                + 0.20
                * math.log1p(max(0, int(popularity.get("recent_magnitude") or 0)))
            )
            rating = row.get("rating", {})
            votes = max(0, int(rating.get("votes") or 0))
            average = float(rating.get("average") or 0.0)
            weighted_rating_sum += average * votes
            rating_votes_sum += votes
        global_rating = weighted_rating_sum / max(1, rating_votes_sum)
        for perfume_id, row in self.rows.items():
            rating = row.get("rating", {})
            votes = max(0, int(rating.get("votes") or 0))
            average = float(rating.get("average") or global_rating)
            quality_raw[perfume_id] = (
                votes * average + 50.0 * global_rating
            ) / (votes + 50.0)
        self.popularity = _rank_percentiles(popularity_raw)
        self.quality = _rank_percentiles(quality_raw)

    @classmethod
    def load(cls, path: Path = DEFAULT_COMMUNITY_PROFILES) -> CommunityFeatureStore:
        return cls(read_jsonl(path))

    def _smoothed_category_fit(
        self,
        perfume_id: str,
        field: str,
        requested: tuple[str, ...],
        priors: dict[str, float],
        alpha: float,
    ) -> float | None:
        if not requested:
            return None
        values = self.rows[perfume_id].get(field, {})
        total = _sum_votes(values)
        fits = [
            (max(0, int(values.get(key, 0))) + alpha * priors.get(key, 0.0))
            / (total + alpha)
            for key in requested
        ]
        return max(fits)

    def season_fit(self, perfume_id: str, seasons: tuple[str, ...]) -> float | None:
        return self._smoothed_category_fit(
            perfume_id, "seasons", seasons, self.season_priors, 40.0
        )

    def raw_season_share(self, perfume_id: str, seasons: tuple[str, ...]) -> float:
        if not seasons:
            return 1.0
        values = self.rows[perfume_id].get("seasons", {})
        total = _sum_votes(values)
        if total == 0:
            return 1.0
        return max(max(0, int(values.get(key, 0))) / total for key in seasons)

    @staticmethod
    def _requested_dayparts(intent: QueryIntent) -> tuple[str, ...]:
        day = {"office", "everyday", "sport", "vacation"}
        night = {"evening", "date", "formal"}
        requested = []
        if day.intersection(intent.occasions):
            requested.append("day")
        if night.intersection(intent.occasions):
            requested.append("night")
        return tuple(requested)

    def daypart_fit(self, perfume_id: str, intent: QueryIntent) -> float | None:
        return self._smoothed_category_fit(
            perfume_id,
            "daypart",
            self._requested_dayparts(intent),
            self.daypart_priors,
            30.0,
        )

    def longevity_fit(self, perfume_id: str, intent: QueryIntent) -> float | None:
        if intent.longevity == "any":
            return None
        average = float(
            self.rows[perfume_id].get("longevity", {}).get("average") or 3.0
        )
        high = min(1.0, max(0.0, (average - 1.0) / 4.0))
        return high if intent.longevity == "high" else 1.0 - high

    def sillage_fit(self, perfume_id: str, intent: QueryIntent) -> float | None:
        if intent.sillage == "any":
            return None
        average = float(
            self.rows[perfume_id].get("sillage", {}).get("average") or 2.5
        )
        strong = min(1.0, max(0.0, (average - 1.0) / 3.0))
        return strong if intent.sillage == "strong" else 1.0 - strong

    def sweetness_fit(self, perfume_id: str, intent: QueryIntent) -> float | None:
        if intent.sweetness == "any":
            return None
        strength = float(
            self.rows[perfume_id].get("accord_strengths", {}).get("sweet", 0.0)
        )
        return strength if intent.sweetness == "high" else 1.0 - strength

    def community_gender_fit(
        self, perfume_id: str, intent: QueryIntent
    ) -> float | None:
        if intent.gender == "any":
            return None
        values = self.rows[perfume_id].get("community_gender", {})
        total = _sum_votes(values)
        if total == 0:
            return 0.5
        if intent.gender == "men":
            score = (
                values.get("male", 0)
                + 0.75 * values.get("male_leaning", 0)
                + 0.5 * values.get("unisex", 0)
                + 0.25 * values.get("female_leaning", 0)
            )
        elif intent.gender == "women":
            score = (
                values.get("female", 0)
                + 0.75 * values.get("female_leaning", 0)
                + 0.5 * values.get("unisex", 0)
                + 0.25 * values.get("male_leaning", 0)
            )
        else:
            score = values.get("unisex", 0) + 0.5 * (
                values.get("male_leaning", 0) + values.get("female_leaning", 0)
            )
        return min(1.0, max(0.0, score / total))

    def reference_graph_scores(
        self, reference_id: str, candidate_id: str
    ) -> tuple[float, float]:
        reference = self.rows.get(reference_id, {})
        reminds = 0.0
        for edge in reference.get("similarity", {}).get("reminds_me_of", []):
            if str(edge["perfume_id"]) == candidate_id:
                reminds = _combine_evidence(
                    reminds,
                    _edge_agreement(edge["up_votes"], edge["down_votes"]),
                )
        for source_id, score in self._reverse_reminds.get(reference_id, []):
            if source_id == candidate_id:
                reminds = _combine_evidence(reminds, 0.85 * score)

        also_liked = 0.0
        for rank, target_id in enumerate(
            reference.get("similarity", {}).get("also_liked", []), start=1
        ):
            if str(target_id) == candidate_id:
                also_liked = math.exp(-(rank - 1) / 8.0)
                break
        for source_id, score in self._reverse_likes.get(reference_id, []):
            if source_id == candidate_id:
                also_liked = max(also_liked, 0.75 * score)
        return reminds, also_liked

    def features(
        self,
        perfume_id: str,
        intent: QueryIntent,
        reference_id: str | None,
    ) -> CommunityCandidateFeatures:
        reminds: float | None = None
        also_liked: float | None = None
        if reference_id is not None:
            reminds, also_liked = self.reference_graph_scores(
                reference_id, perfume_id
            )
        return CommunityCandidateFeatures(
            season_fit=self.season_fit(perfume_id, intent.seasons),
            daypart_fit=self.daypart_fit(perfume_id, intent),
            longevity_fit=self.longevity_fit(perfume_id, intent),
            sillage_fit=self.sillage_fit(perfume_id, intent),
            sweetness_fit=self.sweetness_fit(perfume_id, intent),
            popularity=self.popularity.get(perfume_id, 0.0),
            quality=self.quality.get(perfume_id, 0.0),
            community_gender_fit=self.community_gender_fit(perfume_id, intent),
            reminds_me_of=reminds,
            also_liked=also_liked,
        )

    def violates_community_gate(
        self, perfume_id: str, intent: QueryIntent
    ) -> str | None:
        if intent.seasons and self.raw_season_share(perfume_id, intent.seasons) < 0.10:
            values = self.rows[perfume_id].get("seasons", {})
            if _sum_votes(values) >= 30:
                return "season_mismatch"
        if intent.sweetness == "low":
            sweet = float(
                self.rows[perfume_id]
                .get("accord_strengths", {})
                .get("sweet", 0.0)
            )
            if sweet > 0.55:
                return "too_sweet"
        if intent.popularity == "high" and self.popularity.get(perfume_id, 0.0) < 0.25:
            return "low_popularity"
        if (
            intent.availability == "prefer_common"
            and self.popularity.get(perfume_id, 0.0) < 0.35
        ):
            return "uncommon_proxy"
        return None

    def consensus(self, perfume_id: str) -> dict[str, list[dict[str, Any]]]:
        return self.rows[perfume_id].get("community_summary", {})


def query_dependent_weights(
    intent: QueryIntent, has_reference: bool
) -> dict[str, float]:
    """Return transparent weights; values are normalized after active axes are added."""
    if has_reference:
        weights = {
            "semantic_query": 0.10,
            "reference_content": 0.34,
            "reminds_me_of": 0.34,
            "also_liked": 0.14,
            "link_prediction": 0.10,
            "quality": 0.08,
        }
    else:
        weights = {"semantic_query": 0.82, "quality": 0.18}
    optional = {
        "season_fit": (0.35, bool(intent.seasons)),
        "daypart_fit": (0.18, bool(intent.occasions)),
        "longevity_fit": (0.20, intent.longevity != "any"),
        "sillage_fit": (0.20, intent.sillage != "any"),
        "sweetness_fit": (0.20, intent.sweetness != "any"),
        "structured_fit": (
            0.22,
            bool(intent.preferred_notes or intent.preferred_accords),
        ),
        "popularity": (0.30, intent.popularity == "high"),
        "community_gender_fit": (0.08, intent.gender != "any"),
    }
    for name, (weight, active) in optional.items():
        if active:
            weights[name] = weight
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def weighted_score(
    components: dict[str, float | None], weights: dict[str, float]
) -> tuple[float, dict[str, float]]:
    used = {
        key: weight
        for key, weight in weights.items()
        if components.get(key) is not None
    }
    total = sum(used.values())
    normalized = {key: value / total for key, value in used.items()}
    score = sum(float(components[key]) * weight for key, weight in normalized.items())
    return score, normalized
