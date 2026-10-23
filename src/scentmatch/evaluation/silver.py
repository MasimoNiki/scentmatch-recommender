"""Create deterministic silver relevance labels from structured constraints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl


def _values(profile: dict[str, Any]) -> tuple[set[str], set[str]]:
    notes = {
        str(note).casefold()
        for layer in profile["notes"].values()
        for note in layer
    }
    accords = {str(value).casefold() for value in profile["accords"]}
    return notes, accords


def silver_relevance(query: dict[str, Any], profile: dict[str, Any]) -> int:
    """Assign a 0–3 proxy grade using only explicit query constraints."""
    constraints = query["constraints"]
    notes, accords = _values(profile)
    required_notes = {value.casefold() for value in constraints["required_notes"]}
    preferred_notes = {
        value.casefold() for value in constraints["preferred_notes"]
    }
    excluded_notes = {value.casefold() for value in constraints["excluded_notes"]}
    preferred_accords = {
        value.casefold() for value in constraints["preferred_accords"]
    }
    excluded_accords = {
        value.casefold() for value in constraints["excluded_accords"]
    }

    if not required_notes.issubset(notes):
        return 0
    if notes & excluded_notes or accords & excluded_accords:
        return 0

    requested_gender = constraints["gender"]
    actual_gender = str(profile.get("gender", "")).casefold()
    gender_factor = 1.0
    if requested_gender == "unisex" and actual_gender != "unisex":
        return 0
    if requested_gender in {"women", "men"}:
        if actual_gender == "unisex":
            gender_factor = 0.75
        elif actual_gender != requested_gender:
            return 0

    matched_weight = 2.0 * len(required_notes)
    total_weight = 2.0 * len(required_notes)
    matched_weight += 2.0 * len(preferred_notes & notes)
    total_weight += 2.0 * len(preferred_notes)
    matched_weight += 1.0 * len(preferred_accords & accords)
    total_weight += 1.0 * len(preferred_accords)
    coverage = matched_weight / total_weight if total_weight else 0.0
    coverage *= gender_factor

    if coverage >= 0.7:
        return 3
    if coverage >= 0.35:
        return 2
    if matched_weight > 0:
        return 1
    return 0


def build_silver_qrels(
    queries: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label every query/profile pair and store only positive proxy grades."""
    qrels = []
    per_query: dict[str, dict[str, int]] = {}
    total = Counter()
    for query in queries:
        labels: dict[str, int] = {}
        counts = Counter()
        for profile in profiles:
            grade = silver_relevance(query, profile)
            counts[grade] += 1
            total[grade] += 1
            if grade > 0:
                labels[str(profile["perfume_id"])] = grade
        query_id = query["query_id"]
        qrels.append(
            {
                "query_id": query_id,
                "label_source": "automated_constraints_v1",
                "labels": labels,
            }
        )
        per_query[query_id] = {str(grade): counts[grade] for grade in range(4)}

    warnings = [
        query_id
        for query_id, counts in per_query.items()
        if counts["3"] == 0
    ]
    report = {
        "status": "passed" if not warnings else "review_recommended",
        "label_source": "silver / automated constraints, not human ground truth",
        "queries": len(queries),
        "profiles": len(profiles),
        "pair_count": len(queries) * len(profiles),
        "total_grade_distribution": {
            str(grade): total[grade] for grade in range(4)
        },
        "per_query_grade_distribution": per_query,
        "queries_without_grade_3": warnings,
        "limitations": [
            "style_tags are not automatically evaluated",
            "note positions are not evaluated separately",
            "labels measure metadata constraint matching, not human preference",
        ],
    }
    return qrels, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("data/eval/queries_v1.jsonl")
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/processed/perfume_profiles_5000.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/silver_qrels_v1.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/interim/silver_qrels_report.json")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queries = read_jsonl(args.queries.resolve())
    profiles = read_jsonl(args.profiles.resolve())
    qrels, report = build_silver_qrels(queries, profiles)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in qrels:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Silver relevance labels created.")
    print(f"Queries: {report['queries']}")
    print(f"Evaluated pairs: {report['pair_count']}")
    print(f"Queries without grade 3: {len(report['queries_without_grade_3'])}")
    print(f"Qrels: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
