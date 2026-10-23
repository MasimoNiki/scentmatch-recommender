"""Compare baseline and fine-tuned rankings on the fixed evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scentmatch.data.audit import read_jsonl
from scentmatch.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


def _ranking_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in read_jsonl(path)}


def _qrels(path: Path) -> dict[str, dict[str, int]]:
    return {
        row["query_id"]: {key: int(value) for key, value in row["labels"].items()}
        for row in read_jsonl(path)
    }


def compare_rankings(
    queries: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    finetuned: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-query changes and a compact aggregate report."""
    rows = []
    category_deltas: dict[str, list[float]] = defaultdict(list)
    outcomes = defaultdict(int)
    for query in queries:
        query_id = query["query_id"]
        labels = qrels[query_id]
        baseline_results = baseline[query_id]["results"]
        finetuned_results = finetuned[query_id]["results"]
        baseline_ids = [row["perfume_id"] for row in baseline_results]
        finetuned_ids = [row["perfume_id"] for row in finetuned_results]
        baseline_ndcg = ndcg_at_k(baseline_ids, labels, 10)
        finetuned_ndcg = ndcg_at_k(finetuned_ids, labels, 10)
        delta = finetuned_ndcg - baseline_ndcg
        if delta > 1e-9:
            outcome = "improved"
        elif delta < -1e-9:
            outcome = "regressed"
        else:
            outcome = "unchanged"
        outcomes[outcome] += 1
        category_deltas[query["category"]].append(delta)
        rows.append(
            {
                "query_id": query_id,
                "split": query["split"],
                "category": query["category"],
                "query_text": query["query_text"],
                "baseline_top1": (
                    f"{baseline_results[0]['name']} — {baseline_results[0]['brand']}"
                ),
                "baseline_top1_grade": baseline_results[0]["silver_relevance"],
                "finetuned_top1": (
                    f"{finetuned_results[0]['name']} — {finetuned_results[0]['brand']}"
                ),
                "finetuned_top1_grade": finetuned_results[0]["silver_relevance"],
                "baseline_precision_at_5": precision_at_k(
                    baseline_ids, labels, 5
                ),
                "finetuned_precision_at_5": precision_at_k(
                    finetuned_ids, labels, 5
                ),
                "baseline_mrr_at_10": reciprocal_rank(baseline_ids, labels, 10),
                "finetuned_mrr_at_10": reciprocal_rank(finetuned_ids, labels, 10),
                "baseline_ndcg_at_10": baseline_ndcg,
                "finetuned_ndcg_at_10": finetuned_ndcg,
                "ndcg_delta": delta,
                "outcome": outcome,
            }
        )
    report = {
        "queries": len(rows),
        "outcomes_by_ndcg_at_10": dict(outcomes),
        "mean_ndcg_delta": mean(row["ndcg_delta"] for row in rows),
        "mean_ndcg_delta_by_category": {
            category: mean(values)
            for category, values in sorted(category_deltas.items())
        },
        "largest_improvements": [
            row["query_id"]
            for row in sorted(rows, key=lambda value: value["ndcg_delta"], reverse=True)[
                :5
            ]
        ],
        "largest_regressions": [
            row["query_id"]
            for row in sorted(rows, key=lambda value: value["ndcg_delta"])[:5]
        ],
    }
    return rows, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("data/eval/queries_v1.jsonl")
    )
    parser.add_argument(
        "--qrels", type=Path, default=Path("data/eval/silver_qrels_v1.jsonl")
    )
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/e5_small/rankings.jsonl")
    )
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=Path("artifacts/e5_finetuned/rankings.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/interim/model_comparison.json")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("data/interim/model_comparison.csv")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, report = compare_rankings(
        read_jsonl(args.queries.resolve()),
        _ranking_rows(args.baseline.resolve()),
        _ranking_rows(args.finetuned.resolve()),
        _qrels(args.qrels.resolve()),
    )
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.csv.resolve()
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Model comparison completed.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
