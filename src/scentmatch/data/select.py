"""Select a reproducible 5,000-perfume catalog and write perfume profiles."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scentmatch.data.fragrantica import build_catalog, build_retrieval_text


def proportional_quotas(counts: pd.Series, target_size: int) -> pd.Series:
    """Allocate integer quotas with the largest-remainder method."""
    counts = counts.astype(int).sort_index()
    if target_size < 1:
        raise ValueError("target_size must be positive")
    if target_size > int(counts.sum()):
        raise ValueError("target_size is larger than the eligible pool")
    exact = counts / counts.sum() * target_size
    quotas = exact.map(math.floor).astype(int)
    remainder = target_size - int(quotas.sum())
    order = (exact - quotas).sort_values(ascending=False, kind="stable").index
    for key in order[:remainder]:
        quotas.loc[key] += 1
    return quotas


def _quality_order(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["rating_count", "rating_value", "description_length", "perfume_id"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="stable",
    )


def select_catalog(
    catalog: pd.DataFrame,
    target_size: int = 5_000,
    min_rating_count: int = 100,
    min_description_length: int = 80,
    brand_cap: int = 50,
    rare_accord_threshold: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dedupe, filter, stratify and quality-rank the perfume catalog."""
    if brand_cap < 1:
        raise ValueError("brand_cap must be positive")

    deduplicated = _quality_order(catalog).drop_duplicates(
        "duplicate_key", keep="first"
    )
    full_pyramid = deduplicated[
        ["top_notes", "middle_notes", "base_notes"]
    ].apply(lambda column: column.map(bool)).all(axis=1)
    eligible = deduplicated.loc[
        deduplicated["rating_count"].ge(min_rating_count)
        & deduplicated["description_length"].ge(min_description_length)
        & full_pyramid
    ].copy()
    if len(eligible) < target_size:
        raise ValueError(
            f"Only {len(eligible)} eligible rows for target_size={target_size}"
        )

    accord_counts = eligible["primary_accord"].value_counts()
    common_accords = set(
        accord_counts[accord_counts.ge(rare_accord_threshold)].index
    )
    eligible["_accord_group"] = eligible["primary_accord"].where(
        eligible["primary_accord"].isin(common_accords), "other"
    )
    eligible["_gender_group"] = eligible["gender"].fillna("unknown")
    eligible["_stratum"] = (
        eligible["_gender_group"].astype(str)
        + " | "
        + eligible["_accord_group"].astype(str)
    )

    quotas = proportional_quotas(eligible["_stratum"].value_counts(), target_size)
    ordered = _quality_order(eligible)
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    brand_counts: Counter[str] = Counter()
    shortfalls: dict[str, int] = {}

    for stratum, quota in quotas.items():
        chosen = 0
        rows = ordered.loc[ordered["_stratum"].eq(stratum)]
        for index, row in rows.iterrows():
            brand_key = str(row["brand"]).casefold()
            if brand_counts[brand_key] >= brand_cap:
                continue
            selected_indices.append(index)
            selected_set.add(index)
            brand_counts[brand_key] += 1
            chosen += 1
            if chosen == int(quota):
                break
        if chosen < int(quota):
            shortfalls[str(stratum)] = int(quota) - chosen

    # Brand caps can prevent a few strata from reaching their quota. Fill those
    # places from the best remaining candidates while retaining the same cap.
    for index, row in ordered.iterrows():
        if len(selected_indices) == target_size:
            break
        if index in selected_set:
            continue
        brand_key = str(row["brand"]).casefold()
        if brand_counts[brand_key] >= brand_cap:
            continue
        selected_indices.append(index)
        selected_set.add(index)
        brand_counts[brand_key] += 1

    if len(selected_indices) != target_size:
        raise RuntimeError(
            f"Brand cap left only {len(selected_indices)} selectable rows"
        )

    selected = _quality_order(eligible.loc[selected_indices]).reset_index(drop=True)
    report: dict[str, Any] = {
        "rules": {
            "target_size": target_size,
            "duplicate_key": "normalized brand + name + year",
            "min_rating_count": min_rating_count,
            "min_description_length": min_description_length,
            "require_all_note_layers": True,
            "brand_cap": brand_cap,
            "rare_accord_threshold_for_stratification": rare_accord_threshold,
            "stratification": "gender + primary accord group",
            "ranking": [
                "rating_count descending",
                "rating_value descending",
                "description_length descending",
                "perfume_id ascending",
            ],
        },
        "counts": {
            "joined": len(catalog),
            "deduplicated": len(deduplicated),
            "eligible": len(eligible),
            "selected": len(selected),
            "selected_brands": int(selected["brand"].nunique()),
        },
        "selected_rating_count": {
            "min": int(selected["rating_count"].min()),
            "median": float(selected["rating_count"].median()),
            "max": int(selected["rating_count"].max()),
        },
        "selected_gender": {
            str(key): int(value)
            for key, value in selected["gender"].value_counts().items()
        },
        "selected_primary_accord": {
            str(key): int(value)
            for key, value in selected["primary_accord"].value_counts().items()
        },
        "max_selected_per_brand": max(brand_counts.values()),
        "quota_shortfalls_before_global_fill": shortfalls,
    }
    return selected, report


def _nullable(value: object) -> object:
    return None if pd.isna(value) else value


def row_to_profile(row: pd.Series) -> dict[str, Any]:
    """Convert one selected row to the public perfume_profile schema."""
    year = _nullable(row["year"])
    rating_value = _nullable(row["rating_value"])
    return {
        "schema_version": "1.0",
        "perfume_id": str(row["perfume_id"]),
        "name": row["name"],
        "brand": row["brand"],
        "source_url": row["url"],
        "gender": _nullable(row["gender"]),
        "country": _nullable(row["country"]),
        "year": int(year) if year is not None else None,
        "rating": {
            "value": float(rating_value) if rating_value is not None else None,
            "count": int(row["rating_count"]),
        },
        "perfumers": row["perfumers"],
        "accords": row["accords"],
        "notes": {
            "top": row["top_notes"],
            "middle": row["middle_notes"],
            "base": row["base_notes"],
        },
        "source_description": row["source_description"],
        "retrieval_text": build_retrieval_text(row),
    }


def write_outputs(
    selected: pd.DataFrame, report: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [row_to_profile(row) for _, row in selected.iterrows()]
    suffix = len(profiles)
    jsonl_path = output_dir / f"perfume_profiles_{suffix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for profile in profiles:
            stream.write(json.dumps(profile, ensure_ascii=False) + "\n")

    flat_rows = []
    for profile in profiles:
        flat_rows.append(
            {
                "perfume_id": profile["perfume_id"],
                "name": profile["name"],
                "brand": profile["brand"],
                "gender": profile["gender"],
                "country": profile["country"],
                "year": profile["year"],
                "rating_value": profile["rating"]["value"],
                "rating_count": profile["rating"]["count"],
                "perfumers": json.dumps(profile["perfumers"], ensure_ascii=False),
                "accords": json.dumps(profile["accords"], ensure_ascii=False),
                "top_notes": json.dumps(profile["notes"]["top"], ensure_ascii=False),
                "middle_notes": json.dumps(
                    profile["notes"]["middle"], ensure_ascii=False
                ),
                "base_notes": json.dumps(
                    profile["notes"]["base"], ensure_ascii=False
                ),
                "source_description": profile["source_description"],
                "retrieval_text": profile["retrieval_text"],
                "source_url": profile["source_url"],
            }
        )
    pd.DataFrame(flat_rows).to_csv(
        output_dir / f"perfume_catalog_{suffix}.csv", index=False, encoding="utf-8"
    )
    report_path = output_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return jsonl_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=5_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog, _ = build_catalog(args.raw_dir.resolve())
    selected, report = select_catalog(catalog, target_size=args.size)
    output_dir = args.output_dir.resolve()
    profiles_path, report_path = write_outputs(selected, report, output_dir)
    print("Selection completed.")
    print(f"Selected profiles: {len(selected)}")
    print(f"Eligible pool: {report['counts']['eligible']}")
    print(f"Profiles: {profiles_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
