"""Validate versioned retrieval queries before relevance annotation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl

ALLOWED_SPLITS = {"dev", "test"}
ALLOWED_CATEGORIES = {
    "accord",
    "composition",
    "negation",
    "note",
    "occasion",
    "season",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_GENDERS = {"any", "women", "men", "unisex"}
CONSTRAINT_LIST_FIELDS = {
    "required_notes",
    "preferred_notes",
    "excluded_notes",
    "preferred_accords",
    "excluded_accords",
    "style_tags",
}
REQUIRED_FIELDS = {
    "schema_version",
    "query_id",
    "query_text",
    "language",
    "split",
    "category",
    "difficulty",
    "constraints",
    "annotation_status",
}


def _profile_vocabulary(profiles: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    notes = {
        str(note).casefold()
        for profile in profiles
        for values in profile["notes"].values()
        for note in values
    }
    accords = {
        str(accord).casefold()
        for profile in profiles
        for accord in profile["accords"]
    }
    return notes, accords


def validate_queries(
    queries: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    expected_count: int = 40,
) -> dict[str, Any]:
    """Check query structure, uniqueness and controlled-vocabulary terms."""
    errors: dict[str, list[str]] = defaultdict(list)
    query_ids: list[str] = []
    query_texts: list[str] = []
    known_notes, known_accords = _profile_vocabulary(profiles)

    if len(queries) != expected_count:
        errors["unexpected_query_count"].append(
            f"expected {expected_count}, found {len(queries)}"
        )

    for position, query in enumerate(queries, start=1):
        query_id = str(query.get("query_id", f"line:{position}"))
        query_ids.append(query_id)
        missing = REQUIRED_FIELDS - set(query)
        if missing:
            errors["missing_fields"].append(
                f"{query_id}: {', '.join(sorted(missing))}"
            )
            continue
        if not re.fullmatch(r"q\d{3}", query_id):
            errors["invalid_query_id"].append(query_id)
        text = str(query["query_text"]).strip()
        query_texts.append(text.casefold())
        if len(text) < 20:
            errors["query_text_too_short"].append(query_id)
        if query["language"] != "ru":
            errors["unsupported_language"].append(query_id)
        if query["split"] not in ALLOWED_SPLITS:
            errors["invalid_split"].append(query_id)
        if query["category"] not in ALLOWED_CATEGORIES:
            errors["invalid_category"].append(query_id)
        if query["difficulty"] not in ALLOWED_DIFFICULTIES:
            errors["invalid_difficulty"].append(query_id)
        if query["annotation_status"] != "unreviewed":
            errors["invalid_annotation_status"].append(query_id)

        constraints = query["constraints"]
        if not isinstance(constraints, dict):
            errors["invalid_constraints"].append(query_id)
            continue
        if constraints.get("gender") not in ALLOWED_GENDERS:
            errors["invalid_gender_constraint"].append(query_id)
        for field in CONSTRAINT_LIST_FIELDS:
            values = constraints.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                errors[f"invalid_{field}"].append(query_id)

        required = {value.casefold() for value in constraints["required_notes"]}
        preferred = {value.casefold() for value in constraints["preferred_notes"]}
        excluded = {value.casefold() for value in constraints["excluded_notes"]}
        if (required | preferred) & excluded:
            errors["overlapping_note_constraints"].append(query_id)
        preferred_accords = {
            value.casefold() for value in constraints["preferred_accords"]
        }
        excluded_accords = {
            value.casefold() for value in constraints["excluded_accords"]
        }
        if preferred_accords & excluded_accords:
            errors["overlapping_accord_constraints"].append(query_id)
        for value in required | preferred | excluded:
            if value not in known_notes:
                errors["unknown_note_terms"].append(f"{query_id}: {value}")
        for value in preferred_accords | excluded_accords:
            if value not in known_accords:
                errors["unknown_accord_terms"].append(f"{query_id}: {value}")

    duplicate_ids = [
        key for key, count in Counter(query_ids).items() if count > 1
    ]
    duplicate_texts = [
        key for key, count in Counter(query_texts).items() if count > 1
    ]
    if duplicate_ids:
        errors["duplicate_query_ids"].extend(duplicate_ids)
    if duplicate_texts:
        errors["duplicate_query_texts"].extend(duplicate_texts)

    return {
        "status": "passed" if not errors else "failed",
        "query_count": len(queries),
        "unique_query_ids": len(set(query_ids)),
        "profiles_used_for_vocabulary": len(profiles),
        "distributions": {
            "split": dict(Counter(query["split"] for query in queries)),
            "category": dict(Counter(query["category"] for query in queries)),
            "difficulty": dict(
                Counter(query["difficulty"] for query in queries)
            ),
        },
        "errors": dict(sorted(errors.items())),
        "annotation": {
            "status": "not_started",
            "judgment_scale": [0, 1, 2, 3],
            "note": "Queries are validated, but relevance labels do not exist yet.",
        },
    }


def write_review_csv(queries: list[dict[str, Any]], path: Path) -> None:
    """Write a flat, human-reviewable copy without changing the JSONL source."""
    fields = [
        "query_id",
        "query_text",
        "split",
        "category",
        "difficulty",
        "gender",
        "required_notes",
        "preferred_notes",
        "excluded_notes",
        "preferred_accords",
        "excluded_accords",
        "style_tags",
        "approved",
        "reviewer_comment",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for query in queries:
            constraints = query["constraints"]
            row = {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "split": query["split"],
                "category": query["category"],
                "difficulty": query["difficulty"],
                "gender": constraints["gender"],
                "approved": "",
                "reviewer_comment": "",
            }
            for field in CONSTRAINT_LIST_FIELDS:
                row[field] = json.dumps(constraints[field], ensure_ascii=False)
            writer.writerow(row)


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
        "--report", type=Path, default=Path("data/interim/eval_validation.json")
    )
    parser.add_argument("--expected-count", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queries = read_jsonl(args.queries.resolve())
    profiles = read_jsonl(args.profiles.resolve())
    report = validate_queries(queries, profiles, args.expected_count)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_path = report_path.parent / "eval_queries_review.csv"
    write_review_csv(queries, review_path)
    print(f"Evaluation query validation {report['status']}.")
    print(f"Queries: {report['query_count']}")
    print(f"Unique query IDs: {report['unique_query_ids']}")
    print("Relevance judgments: not started")
    print(f"Report: {report_path}")
    print(f"Review CSV: {review_path}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
