"""Build compact community profiles from the Le Decanteur perfume dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl

DEFAULT_PROFILES = Path("data/processed/perfume_profiles_5000.jsonl")
DEFAULT_OUTPUT = Path("data/augmentation/community_profiles_v2.jsonl")
DEFAULT_REPORT = Path("data/interim/community_profiles_v2_report.json")
SOURCE_NAME = "Le Decanteur — Fragrantica Perfumes v2"
SOURCE_URL = "https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes"
SOURCE_LICENSE = "CC BY-NC-SA 4.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _histogram_votes(value: dict[str, Any] | None) -> int:
    if not value:
        return 0
    return sum(int(row.get("count", 0)) for row in value.get("histogram", []))


def _agreement_evidence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep auditable consensus claims; Wilson score prevents tiny samples winning."""
    result = []
    for row in rows:
        up = max(0, int(row.get("up_votes", 0)))
        down = max(0, int(row.get("down_votes", 0)))
        votes = up + down
        if votes < 5:
            continue
        probability = up / votes
        z = 1.96
        denominator = 1.0 + z * z / votes
        centre = probability + z * z / (2.0 * votes)
        spread = z * math.sqrt(
            probability * (1.0 - probability) / votes
            + z * z / (4.0 * votes * votes)
        )
        lower_bound = (centre - spread) / denominator
        if probability < 0.60:
            continue
        result.append(
            {
                "text": str(row.get("text", "")).strip(),
                "up_votes": up,
                "down_votes": down,
                "agreement": round(probability, 4),
                "wilson_lower": round(lower_bound, 4),
            }
        )
    return sorted(
        result,
        key=lambda item: (item["wilson_lower"], item["up_votes"]),
        reverse=True,
    )[:6]


def compact_community_profile(
    row: dict[str, Any], catalog_ids: set[str]
) -> dict[str, Any]:
    similar = row.get("similar") or {}
    reminds = []
    for edge in similar.get("reminds_me_of", []):
        target_id = str(edge.get("id", ""))
        if target_id not in catalog_ids:
            continue
        reminds.append(
            {
                "perfume_id": target_id,
                "up_votes": max(0, int(edge.get("up_votes", 0))),
                "down_votes": max(0, int(edge.get("down_votes", 0))),
            }
        )
    also_liked = [
        str(edge.get("id"))
        for edge in similar.get("also_liked", [])
        if str(edge.get("id")) in catalog_ids
    ]
    accords = {
        str(item.get("name", "")).casefold(): round(
            float(item.get("strength", 0)) / 100.0, 4
        )
        for item in row.get("accords", [])
        if item.get("name")
    }
    summary = row.get("ai_summary") or {}
    rating = row.get("rating") or {}
    longevity = row.get("longevity") or {}
    sillage = row.get("sillage") or {}
    return {
        "perfume_id": str(row["id"]),
        "name": row.get("name", ""),
        "brand": row.get("brand", ""),
        "accord_strengths": accords,
        "rating": {
            "average": rating.get("average"),
            "votes": _histogram_votes(rating),
        },
        "longevity": {
            "average": longevity.get("average"),
            "votes": _histogram_votes(longevity),
        },
        "sillage": {
            "average": sillage.get("average"),
            "votes": _histogram_votes(sillage),
        },
        "relation": row.get("relation") or {},
        "community_gender": row.get("community_gender") or {},
        "seasons": row.get("seasons") or {},
        "daypart": row.get("daypart") or {},
        "people": int(row.get("people") or 0),
        "popularity": row.get("popularity") or {},
        "community_summary": {
            "pros": _agreement_evidence(summary.get("pros", [])),
            "cons": _agreement_evidence(summary.get("cons", [])),
            "warning": "Source-provided AI summary, retained only with user votes.",
        },
        "similarity": {
            "reminds_me_of": reminds,
            "also_liked": also_liked,
        },
        "source": {
            "name": SOURCE_NAME,
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "scraped_at": (row.get("meta") or {}).get("scraped_at"),
        },
    }


def build_community_profiles(
    source_path: Path, profiles_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles = read_jsonl(profiles_path)
    catalog_ids = {str(profile["perfume_id"]) for profile in profiles}
    compact = []
    source_rows = 0
    with source_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            if str(row.get("id")) in catalog_ids:
                compact.append(compact_community_profile(row, catalog_ids))
    compact.sort(key=lambda item: int(item["perfume_id"]))
    matched_ids = {row["perfume_id"] for row in compact}
    report = {
        "schema_version": "2.0",
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_license": SOURCE_LICENSE,
        "source_rows": source_rows,
        "catalog_profiles": len(catalog_ids),
        "matched_profiles": len(compact),
        "coverage": round(len(compact) / max(1, len(catalog_ids)), 6),
        "missing_ids": sorted(catalog_ids - matched_ids, key=int),
        "reminds_me_of_edges_in_catalog": sum(
            len(row["similarity"]["reminds_me_of"]) for row in compact
        ),
        "also_liked_edges_in_catalog": sum(
            len(row["similarity"]["also_liked"]) for row in compact
        ),
        "profiles_with_consensus_pros": sum(
            bool(row["community_summary"]["pros"]) for row in compact
        ),
        "profiles_with_consensus_cons": sum(
            bool(row["community_summary"]["cons"]) for row in compact
        ),
        "source_sha256": _sha256(source_path),
        "profiles_sha256": _sha256(profiles_path),
        "redistribution": "Non-commercial; derived data remains CC BY-NC-SA 4.0.",
    }
    return compact, report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows, report = build_community_profiles(
        args.source.resolve(), args.profiles.resolve()
    )
    _write_jsonl(args.output.resolve(), rows)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
