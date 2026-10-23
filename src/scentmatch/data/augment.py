"""Build conservative community-signal and price augmentation files."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from scentmatch.data.audit import read_jsonl
from scentmatch.data.fragrantica import normalize_identifier

DEFAULT_PROFILES = Path("data/processed/perfume_profiles_5000.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/augmentation")
DEFAULT_REPORT = Path("data/interim/augmentation_report.json")


def _parse_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [
        item.strip().strip("'\"")
        for item in text.strip("[]").split(",")
        if item.strip().strip("'\"")
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _profile_index(profiles: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles:
        key = (
            normalize_identifier(profile["brand"]),
            normalize_identifier(profile["name"]),
        )
        index[key].append(profile)
    return index


def _is_small_format(title: str) -> bool:
    lowered = title.casefold()
    if re.search(r"\b(sample|vial|miniature|mini|decant)\b", lowered):
        return True
    ml_sizes = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*ml\b", lowered)]
    oz_sizes = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:fl\s*)?oz\b", lowered)]
    known_sizes = [*ml_sizes, *(value * 29.5735 for value in oz_sizes)]
    return bool(known_sizes) and max(known_sizes) < 30


def build_community_signals(
    aromo_path: Path, profiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = pd.read_csv(aromo_path, sep=";", decimal=",", low_memory=False)
    index = _profile_index(profiles)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source.to_dict(orient="records"):
        key = (
            normalize_identifier(row.get("brand", "")),
            normalize_identifier(row.get("name", "")),
        )
        matches = index.get(key, [])
        if not matches:
            continue
        notes = _parse_list(row.get("users_notes"))
        segment = "" if pd.isna(row.get("segment")) else str(row["segment"]).strip()
        for profile in matches:
            candidates[str(profile["perfume_id"])].append(
                {
                    "perfume_id": str(profile["perfume_id"]),
                    "segment": segment,
                    "community_perceived_notes": notes,
                    "source_url": str(row.get("url", "")),
                    "source": "Aromo.ru fragrance dataset",
                    "match_method": "exact_normalized_brand_and_name",
                }
            )

    output = []
    for rows in candidates.values():
        # Prefer the row with more community information, deterministically.
        chosen = max(
            rows,
            key=lambda row: (
                bool(row["community_perceived_notes"]),
                bool(row["segment"]),
                len(row["community_perceived_notes"]),
                row["source_url"],
            ),
        )
        if chosen["community_perceived_notes"] or chosen["segment"]:
            output.append(chosen)
    return sorted(output, key=lambda row: int(row["perfume_id"]))


def build_price_snapshot(
    ebay_paths: list[Path], profiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    listings = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in ebay_paths],
        ignore_index=True,
    )
    by_brand: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        by_brand[normalize_identifier(profile["brand"])].append(profile)

    matched: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in listings.to_dict(orient="records"):
        brand_key = normalize_identifier(row.get("brand", ""))
        title = str(row.get("title", ""))
        if _is_small_format(title):
            continue
        title_key = normalize_identifier(title)
        possible = []
        for profile in by_brand.get(brand_key, []):
            name_key = normalize_identifier(profile["name"])
            if len(name_key) >= 4 and name_key in title_key:
                possible.append(profile)
        # Ambiguous listings are discarded instead of guessing a variant.
        if len(possible) != 1:
            continue
        price = pd.to_numeric(row.get("price"), errors="coerce")
        if pd.isna(price) or float(price) <= 0:
            continue
        profile = possible[0]
        matched[str(profile["perfume_id"])].append(
            {"price": float(price), "title": title}
        )

    output = []
    for perfume_id, rows in matched.items():
        prices = sorted(row["price"] for row in rows)
        output.append(
            {
                "perfume_id": perfume_id,
                "currency": "USD",
                "median_price": round(float(median(prices)), 2),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2),
                "listing_count": len(rows),
                "example_title": rows[0]["title"],
                "snapshot": "2024-05",
                "source": "eBay perfume e-commerce dataset 2024",
                "match_method": "unique_normalized_brand_and_name_in_title",
                "warning": "Historical marketplace listing price; not a live retail quote.",
            }
        )
    return sorted(output, key=lambda row: int(row["perfume_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aromo", type=Path, required=True)
    parser.add_argument("--ebay", type=Path, nargs="+", required=True)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    profiles = read_jsonl(args.profiles.resolve())
    community = build_community_signals(args.aromo.resolve(), profiles)
    prices = build_price_snapshot([path.resolve() for path in args.ebay], profiles)
    output_dir = args.output_dir.resolve()
    _write_jsonl(output_dir / "community_signals.jsonl", community)
    _write_jsonl(output_dir / "price_snapshot_2024.jsonl", prices)
    report = {
        "profiles": len(profiles),
        "community_signal_matches": len(community),
        "with_community_notes": sum(
            bool(row["community_perceived_notes"]) for row in community
        ),
        "with_segment": sum(bool(row["segment"]) for row in community),
        "price_matches": len(prices),
        "price_quality_filter": "Excluded samples, vials, minis and formats under 30 ml.",
        "matching_policy": "Conservative exact/unique matching; ambiguous rows discarded.",
        "price_warning": "2024 marketplace snapshot, not live prices.",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Augmentation completed.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
