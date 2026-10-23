"""Human-judged ranking metrics for representative ScentMatch evaluation v3."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl

DEFAULT_QUERIES = Path("data/eval/ranking_queries_v3.jsonl")


def dcg(relevances: list[int], k: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def ndcg(relevances: list[int], k: int) -> float:
    ideal = dcg(sorted(relevances, reverse=True), k)
    return dcg(relevances, k) / ideal if ideal else 0.0


def exposure_gini(counts: list[int]) -> float:
    positive = sorted(value for value in counts if value > 0)
    if not positive:
        return 0.0
    total = sum(positive)
    size = len(positive)
    weighted = sum((index + 1) * value for index, value in enumerate(positive))
    return (2.0 * weighted) / (size * total) - (size + 1) / size


def evaluate_ranking_rows(
    queries: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    query_index = {str(row["query_id"]): row for row in queries}
    judgment_index = {
        (str(row["query_id"]), str(row["perfume_id"])): row
        for row in judgments
    }
    per_query = []
    exposures: Counter[str] = Counter()
    brands: dict[str, str] = {}
    missing_judgments = []
    for ranking in rankings:
        query_id = str(ranking["query_id"])
        if query_id not in query_index:
            continue
        results = ranking.get("results", [])[:k]
        relevance = []
        violations = []
        evidence = []
        query_missing = []
        brand_counts: Counter[str] = Counter()
        for result in results:
            perfume_id = str(result["perfume_id"])
            exposures[perfume_id] += 1
            brand = str(result.get("brand", "")).casefold()
            brands[perfume_id] = brand
            brand_counts[brand] += 1
            judgment = judgment_index.get((query_id, perfume_id))
            if judgment is None:
                query_missing.append(perfume_id)
                continue
            relevance.append(int(judgment["relevance"]))
            violations.append(bool(judgment["constraint_violation"]))
            evidence.append(int(judgment["evidence_quality"]))
        if query_missing:
            missing_judgments.append(
                {"query_id": query_id, "perfume_ids": query_missing}
            )
            continue
        per_query.append(
            {
                "query_id": query_id,
                "category": query_index[query_id]["category"],
                "ndcg_at_k": ndcg(relevance, k),
                "precision_at_k": (
                    sum(value >= 2 for value in relevance) / max(1, len(relevance))
                ),
                "constraint_violation_rate": (
                    sum(violations) / max(1, len(violations))
                ),
                "evidence_quality": sum(evidence) / max(1, 2 * len(evidence)),
                "brand_duplicates": sum(
                    max(0, count - 1) for count in brand_counts.values()
                ),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        if not rows:
            return {"queries": 0}
        metric_names = (
            "ndcg_at_k",
            "precision_at_k",
            "constraint_violation_rate",
            "evidence_quality",
            "brand_duplicates",
        )
        output: dict[str, float | int] = {"queries": len(rows)}
        for name in metric_names:
            output[name] = round(
                sum(float(row[name]) for row in rows) / len(rows), 6
            )
        return output

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        categories[row["category"]].append(row)
    total_exposures = sum(exposures.values())
    report = {
        "schema_version": "3.0",
        "k": k,
        "queries_in_design": len(queries),
        "rankings_evaluated": len(per_query),
        "overall": aggregate(per_query),
        "by_category": {
            category: aggregate(rows) for category, rows in sorted(categories.items())
        },
        "catalog_coverage_at_k": len(exposures),
        "exposure_gini": round(exposure_gini(list(exposures.values())), 6),
        "maximum_item_exposure_share": round(
            max(exposures.values(), default=0) / max(1, total_exposures), 6
        ),
        "missing_judgments": missing_judgments,
        "complete": not missing_judgments and bool(per_query),
        "quality_gates": {
            "constraint_violation_rate_lte_0_02": bool(per_query)
            and float(aggregate(per_query)["constraint_violation_rate"]) <= 0.02,
            "ndcg_at_5_gte_0_70": bool(per_query)
            and float(aggregate(per_query)["ndcg_at_k"]) >= 0.70,
            "maximum_item_exposure_share_lte_0_10": bool(per_query)
            and max(exposures.values(), default=0) / max(1, total_exposures) <= 0.10,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate_ranking_rows(
        read_jsonl(args.queries.resolve()),
        read_jsonl(args.rankings.resolve()),
        read_jsonl(args.judgments.resolve()),
        args.k,
    )
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
