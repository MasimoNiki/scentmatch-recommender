"""Evaluate QueryIntent extraction and reference-perfume entity resolution."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl
from scentmatch.retrieval.intent import parse_query_intent
from scentmatch.retrieval.reference import resolve_reference_detailed

DEFAULT_PROFILES = Path("data/processed/perfume_profiles_5000.jsonl")
DEFAULT_INTENT_CASES = Path("data/eval/query_intent_v2.jsonl")
DEFAULT_REFERENCE_CASES = Path("data/eval/reference_resolution_v2.jsonl")
DEFAULT_REPORT = Path("data/interim/evaluation_v2_report.json")


def _normalise(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def evaluate_intents(cases: list[dict[str, Any]]) -> dict[str, Any]:
    field_totals: dict[str, int] = defaultdict(int)
    field_correct: dict[str, int] = defaultdict(int)
    failures = []
    passed_cases = 0
    for case in cases:
        observed = parse_query_intent(case["query_text"]).to_dict()
        case_ok = True
        for field, expected in case["expected"].items():
            field_totals[field] += 1
            actual = observed[field]
            if _normalise(actual) == _normalise(expected):
                field_correct[field] += 1
            else:
                case_ok = False
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        passed_cases += int(case_ok)
    checked_fields = sum(field_totals.values())
    correct_fields = sum(field_correct.values())
    return {
        "cases": len(cases),
        "case_accuracy": passed_cases / len(cases) if cases else 0.0,
        "field_accuracy": correct_fields / checked_fields if checked_fields else 0.0,
        "per_field_accuracy": {
            field: field_correct[field] / total
            for field, total in sorted(field_totals.items())
        },
        "failures": failures,
    }


def evaluate_references(
    cases: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    positive = [case for case in cases if case["should_resolve"]]
    negative = [case for case in cases if not case["should_resolve"]]
    positive_correct = 0
    negative_correct = 0
    failures = []
    for case in cases:
        result = resolve_reference_detailed(case["query_text"], profiles)
        actual_id = (
            str(result.profile["perfume_id"]) if result.profile is not None else None
        )
        if case["should_resolve"]:
            correct = result.accepted and actual_id == case["expected_perfume_id"]
            positive_correct += int(correct)
        else:
            correct = not result.accepted
            negative_correct += int(correct)
        if not correct:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "query_text": case["query_text"],
                    "expected_perfume_id": case["expected_perfume_id"],
                    "actual": result.to_dict(),
                }
            )
    total_correct = positive_correct + negative_correct
    return {
        "cases": len(cases),
        "positive_cases": len(positive),
        "negative_cases": len(negative),
        "top1_accuracy": positive_correct / len(positive) if positive else 0.0,
        "abstain_accuracy": negative_correct / len(negative) if negative else 0.0,
        "overall_accuracy": total_correct / len(cases) if cases else 0.0,
        "failures": failures,
    }


def build_report(
    intent_cases: list[dict[str, Any]],
    reference_cases: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    intent_metrics = evaluate_intents(intent_cases)
    reference_metrics = evaluate_references(reference_cases, profiles)
    gates = {
        "intent_field_accuracy_gte_0_90": intent_metrics["field_accuracy"] >= 0.90,
        "reference_top1_accuracy_gte_0_95": (
            reference_metrics["top1_accuracy"] >= 0.95
        ),
        "reference_abstain_accuracy_eq_1": (
            reference_metrics["abstain_accuracy"] == 1.0
        ),
    }
    return {
        "schema_version": "2.0",
        "evaluation_scope": [
            "query_intent_extraction",
            "reference_entity_resolution",
        ],
        "intent": intent_metrics,
        "reference": reference_metrics,
        "quality_gates": gates,
        "all_quality_gates_passed": all(gates.values()),
        "limitations": [
            "Ranking relevance and seasonal fit require human judgments.",
            "Availability is not proven by rating.count popularity.",
            "Season, occasion, longevity and sillage ranking await community-vote data.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--intent-cases", type=Path, default=DEFAULT_INTENT_CASES)
    parser.add_argument("--reference-cases", type=Path, default=DEFAULT_REFERENCE_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    profiles = read_jsonl(args.profiles.resolve())
    intent_cases = read_jsonl(args.intent_cases.resolve())
    reference_cases = read_jsonl(args.reference_cases.resolve())
    report = build_report(intent_cases, reference_cases, profiles)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
