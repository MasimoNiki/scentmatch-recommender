"""Dependency-light ranked retrieval metrics."""

from __future__ import annotations

import math
from statistics import mean


def recall_at_k(ranking: list[str], labels: dict[str, int], k: int) -> float:
    relevant = {perfume_id for perfume_id, grade in labels.items() if grade >= 2}
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: list[str], labels: dict[str, int], k: int) -> float:
    for position, perfume_id in enumerate(ranking[:k], start=1):
        if labels.get(perfume_id, 0) >= 2:
            return 1.0 / position
    return 0.0


def precision_at_k(ranking: list[str], labels: dict[str, int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    retrieved = ranking[:k]
    relevant_count = sum(labels.get(perfume_id, 0) >= 2 for perfume_id in retrieved)
    return relevant_count / k


def hit_rate_at_k(ranking: list[str], labels: dict[str, int], k: int) -> float:
    return float(any(labels.get(perfume_id, 0) >= 2 for perfume_id in ranking[:k]))


def ndcg_at_k(ranking: list[str], labels: dict[str, int], k: int) -> float:
    def dcg(grades: list[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(position + 2)
            for position, grade in enumerate(grades)
        )

    actual = [labels.get(perfume_id, 0) for perfume_id in ranking[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]]
) -> dict[str, float]:
    query_ids = sorted(set(rankings) & set(qrels))
    if not query_ids:
        raise ValueError("No matching query IDs between rankings and qrels")
    return {
        "queries": float(len(query_ids)),
        "recall_at_5": mean(
            recall_at_k(rankings[key], qrels[key], 5) for key in query_ids
        ),
        "recall_at_10": mean(
            recall_at_k(rankings[key], qrels[key], 10) for key in query_ids
        ),
        "precision_at_5": mean(
            precision_at_k(rankings[key], qrels[key], 5) for key in query_ids
        ),
        "precision_at_10": mean(
            precision_at_k(rankings[key], qrels[key], 10) for key in query_ids
        ),
        "hit_rate_at_5": mean(
            hit_rate_at_k(rankings[key], qrels[key], 5) for key in query_ids
        ),
        "hit_rate_at_10": mean(
            hit_rate_at_k(rankings[key], qrels[key], 10) for key in query_ids
        ),
        "mrr_at_10": mean(
            reciprocal_rank(rankings[key], qrels[key], 10) for key in query_ids
        ),
        "ndcg_at_10": mean(
            ndcg_at_k(rankings[key], qrels[key], 10) for key in query_ids
        ),
    }
