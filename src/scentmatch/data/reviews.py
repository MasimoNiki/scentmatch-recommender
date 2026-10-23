"""Prepare privacy-minimized review chunks matched to the ScentMatch catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl
from scentmatch.data.fragrantica import normalize_identifier

DEFAULT_PROFILES = Path("data/processed/perfume_profiles_5000.jsonl")
DEFAULT_OUTPUT = Path("data/reviews/atrafshan_review_chunks.jsonl")
DEFAULT_REPORT = Path("data/interim/review_dataset_report.json")
SOURCE_URL = (
    "https://github.com/Kalashi-Saed-Collaborations/"
    "SentimentDrivenCommunityDetection"
)


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _scent_vote(row: dict[str, Any]) -> float | None:
    votes = (row.get("user_vote_on_perfume") or {}).get("votes") or {}
    value = votes.get("رایحه")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_reply(row: dict[str, Any]) -> bool:
    value = (row.get("user_vote_on_perfume") or {}).get("is_comment_answer", "")
    return str(value).strip().casefold() == "yes"


def _build_profile_index(
    profiles: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        key = (
            normalize_identifier(profile["brand"]),
            normalize_identifier(profile["name"]),
        )
        index[key].append(profile)
    return index


def prepare_review_chunks(
    source_rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    max_chunks: int = 3,
    reviews_per_chunk: int = 3,
    max_review_chars: int = 1800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _build_profile_index(profiles)
    matched_rows = 0
    eligible_rows = 0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profile_by_id = {str(profile["perfume_id"]): profile for profile in profiles}

    for row in source_rows:
        key = (
            normalize_identifier(row.get("perfume_brand", "")),
            normalize_identifier(row.get("perfume_name", "")),
        )
        matches = index.get(key, [])
        if len(matches) != 1:
            continue
        matched_rows += 1
        text = _clean_text(row.get("comment_text"))
        vote = _scent_vote(row)
        if _is_reply(row) or len(text) < 40 or (vote is not None and vote < 5):
            continue
        eligible_rows += 1
        grouped[str(matches[0]["perfume_id"])].append(
            {"text": text, "scent_vote": vote}
        )

    chunks: list[dict[str, Any]] = []
    perfumes_with_reviews = 0
    for perfume_id, rows in grouped.items():
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            unique.setdefault(normalize_identifier(row["text"]), row)
        ranked = sorted(
            unique.values(),
            key=lambda row: (
                row["scent_vote"] is not None,
                row["scent_vote"] or 0.0,
                min(len(row["text"]), max_review_chars),
            ),
            reverse=True,
        )
        selected = ranked[: max_chunks * reviews_per_chunk]
        if not selected:
            continue
        perfumes_with_reviews += 1
        profile = profile_by_id[perfume_id]
        for chunk_number in range(max_chunks):
            start = chunk_number * reviews_per_chunk
            chunk_reviews = selected[start : start + reviews_per_chunk]
            if not chunk_reviews:
                break
            review_parts = [
                f"User review: {row['text'][:max_review_chars]}"
                for row in chunk_reviews
            ]
            text = (
                f"Reviews for {profile['name']} by {profile['brand']}. "
                + " ".join(review_parts)
            )
            chunks.append(
                {
                    "chunk_id": f"{perfume_id}:review:{chunk_number + 1}",
                    "perfume_id": perfume_id,
                    "name": profile["name"],
                    "brand": profile["brand"],
                    "language": "fa",
                    "text": text,
                    "reviews_in_chunk": len(chunk_reviews),
                    "eligible_reviews_for_perfume": len(ranked),
                    "source": "Atrafshan perfume review dataset",
                    "source_url": SOURCE_URL,
                }
            )

    report = {
        "source_rows": len(source_rows),
        "source_unique_perfumes": len(
            {str(row.get("perfume_id", "")) for row in source_rows}
        ),
        "exactly_matched_review_rows": matched_rows,
        "eligible_standalone_review_rows": eligible_rows,
        "matched_catalog_perfumes": perfumes_with_reviews,
        "review_chunks": len(chunks),
        "language": "Persian (fa)",
        "matching": "unique exact normalized brand+name",
        "privacy": "User IDs and usernames are not retained.",
        "quality_filter": "Replies, texts under 40 chars and known scent votes below 5 are excluded.",
        "source_url": SOURCE_URL,
        "terms": "Academic/non-commercial use with attribution per repository README.",
    }
    return chunks, report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    source_path = args.source.resolve()
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    profiles = read_jsonl(args.profiles.resolve())
    chunks, report = prepare_review_chunks(source_rows, profiles)
    report["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _write_jsonl(args.output.resolve(), chunks)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Review dataset prepared.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
