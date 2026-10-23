"""Validate the downloaded Fragrantica files and write a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scentmatch.data.fragrantica import extract_fragrance_id

CLEAN_COLUMNS = {
    "url",
    "Perfume",
    "Brand",
    "Country",
    "Gender",
    "Rating Value",
    "Rating Count",
    "Year",
    "Top",
    "Middle",
    "Base",
    "Perfumer1",
    "Perfumer2",
    "mainaccord1",
    "mainaccord2",
    "mainaccord3",
    "mainaccord4",
    "mainaccord5",
}
RAW_COLUMNS = {
    "Name",
    "Gender",
    "Rating Value",
    "Rating Count",
    "Main Accords",
    "Perfumers",
    "Description",
    "url",
}
def require_columns(frame: pd.DataFrame, expected: set[str], filename: str) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def validate_dataset(raw_dir: Path) -> dict[str, object]:
    clean_path = raw_dir / "fra_cleaned.csv"
    raw_path = raw_dir / "fra_perfumes.csv"
    for path in (clean_path, raw_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scentmatch-download before validation."
            )

    clean = pd.read_csv(
        clean_path,
        sep=";",
        encoding="cp1250",
        low_memory=False,
    )
    raw = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)
    require_columns(clean, CLEAN_COLUMNS, clean_path.name)
    require_columns(raw, RAW_COLUMNS, raw_path.name)

    full_pyramid = clean[["Top", "Middle", "Base"]].apply(nonempty).all(axis=1)
    clean_ids = clean["url"].map(extract_fragrance_id)
    raw_ids = set(raw["url"].map(extract_fragrance_id).dropna())
    matched = clean_ids.isin(raw_ids)

    report: dict[str, object] = {
        "status": "passed",
        "clean_rows": len(clean),
        "raw_rows": len(raw),
        "clean_exact_duplicate_rows": int(clean.duplicated().sum()),
        "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
        "clean_duplicate_urls": int(clean["url"].duplicated().sum()),
        "raw_duplicate_urls": int(raw["url"].duplicated().sum()),
        "clean_full_pyramid_rows": int(full_pyramid.sum()),
        "clean_full_pyramid_share": float(full_pyramid.mean()),
        "clean_rows_matched_to_raw": int(matched.sum()),
        "clean_match_share": float(matched.mean()),
        "clean_unique_brands": int(clean["Brand"].nunique()),
        "clean_unique_primary_accords": int(clean["mainaccord1"].nunique()),
        "clean_rows_with_year": int(clean["Year"].notna().sum()),
        "raw_rows_with_description": int(raw["Description"].notna().sum()),
    }

    failures: list[str] = []
    if len(clean) < 20_000:
        failures.append("fra_cleaned.csv has fewer than 20,000 rows")
    if len(raw) < 60_000:
        failures.append("fra_perfumes.csv has fewer than 60,000 rows")
    if full_pyramid.mean() < 0.95:
        failures.append("fewer than 95% of clean profiles have all note layers")
    if matched.mean() < 0.95:
        failures.append("fewer than 95% of clean profiles match the raw file")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        raise ValueError("Dataset validation failed: " + "; ".join(failures))

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the source CSV files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/interim/fragrantica_validation.json"),
        help="Destination for the JSON report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_dataset(args.raw_dir.resolve())
    report_path: Path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Dataset validation passed.")
    print(f"Clean rows: {report['clean_rows']}")
    print(f"Raw rows: {report['raw_rows']}")
    print(
        "Clean profiles with all note layers: "
        f"{report['clean_full_pyramid_rows']}"
    )
    print(
        "Clean profiles matched to a raw description: "
        f"{report['clean_rows_matched_to_raw']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
