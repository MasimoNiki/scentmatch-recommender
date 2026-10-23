"""Shared parsing and normalization for the selected Fragrantica dataset."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

NOTE_COLUMNS = ("Top", "Middle", "Base")
ACCORD_COLUMNS = tuple(f"mainaccord{index}" for index in range(1, 6))
UNKNOWN_VALUES = {"", "nan", "none", "null", "unknown"}
FRAGRANCE_ID_PATTERN = re.compile(r"-(\d+)\.html", re.IGNORECASE)


def extract_fragrance_id(value: object) -> str | None:
    """Extract the stable numeric fragrance ID from a Fragrantica URL."""
    match = FRAGRANCE_ID_PATTERN.search(str(value))
    return match.group(1) if match else None


def normalize_identifier(value: object) -> str:
    """Normalize a name for deterministic duplicate detection."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", "", normalized)


def split_values(value: object) -> list[str]:
    """Split a source cell and remove empty values and in-layer repeats."""
    if pd.isna(value):
        return []
    values = [item.strip() for item in str(value).split(",")]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = item.casefold()
        if not item or key in UNKNOWN_VALUES or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def clean_whitespace(value: object) -> str:
    """Collapse source whitespace without rewriting copyrighted text."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_decimal(series: pd.Series) -> pd.Series:
    """Parse decimal values stored with either comma or dot separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_integer(series: pd.Series) -> pd.Series:
    """Parse integer-like values that may contain thousands separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).astype("Int64")


def read_source_files(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two source CSV files with their actual encodings."""
    clean_path = raw_dir / "fra_cleaned.csv"
    raw_path = raw_dir / "fra_perfumes.csv"
    for path in (clean_path, raw_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scentmatch-download before this command."
            )
    clean = pd.read_csv(
        clean_path,
        sep=";",
        encoding="cp1250",
        low_memory=False,
    )
    raw = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)
    return clean, raw


def build_catalog(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join both source files into one normalized, loss-aware catalog."""
    clean, raw = read_source_files(raw_dir)
    clean = clean.copy()
    raw = raw.copy()
    clean["perfume_id"] = clean["url"].map(extract_fragrance_id)
    raw["perfume_id"] = raw["url"].map(extract_fragrance_id)
    raw["_description_length"] = raw["Description"].fillna("").str.len()

    source_stats = {
        "clean_rows": len(clean),
        "raw_rows": len(raw),
        "clean_unique_ids": int(clean["perfume_id"].nunique()),
        "raw_unique_ids": int(raw["perfume_id"].nunique()),
        "raw_duplicate_ids": int(raw["perfume_id"].duplicated().sum()),
    }

    # Duplicate raw IDs are exact or near-exact crawl repeats. Prefer the row
    # with the longest description, then keep the first row deterministically.
    raw_one = (
        raw.sort_values(
            ["perfume_id", "_description_length", "url"],
            ascending=[True, False, True],
        )
        .drop_duplicates("perfume_id", keep="first")
        .loc[:, ["perfume_id", "Description"]]
    )
    joined = clean.merge(raw_one, on="perfume_id", how="left", validate="one_to_one")

    catalog = pd.DataFrame(
        {
            "perfume_id": joined["perfume_id"].astype(str),
            "name": joined["Perfume"].astype(str).str.strip(),
            "brand": joined["Brand"].astype(str).str.strip(),
            "url": joined["url"].astype(str),
            "country": joined["Country"].astype("string").str.strip(),
            "gender": joined["Gender"].astype("string").str.strip(),
            "year": parse_integer(joined["Year"]),
            "rating_value": parse_decimal(joined["Rating Value"]),
            "rating_count": parse_integer(joined["Rating Count"]),
            "top_notes": joined["Top"].map(split_values),
            "middle_notes": joined["Middle"].map(split_values),
            "base_notes": joined["Base"].map(split_values),
            "accords": joined.loc[:, list(ACCORD_COLUMNS)].apply(
                lambda row: [
                    str(value).strip()
                    for value in row
                    if pd.notna(value)
                    and str(value).strip().casefold() not in UNKNOWN_VALUES
                ],
                axis=1,
            ),
            "perfumers": joined.loc[:, ["Perfumer1", "Perfumer2"]].apply(
                lambda row: [
                    str(value).strip()
                    for value in row
                    if pd.notna(value)
                    and str(value).strip().casefold() not in UNKNOWN_VALUES
                ],
                axis=1,
            ),
            "source_description": joined["Description"].map(clean_whitespace),
        }
    )
    catalog["primary_accord"] = catalog["accords"].map(
        lambda values: values[0] if values else "unknown"
    )
    catalog["description_length"] = catalog["source_description"].str.len()
    catalog["note_count"] = catalog[
        ["top_notes", "middle_notes", "base_notes"]
    ].apply(lambda row: sum(len(values) for values in row), axis=1)
    year_key = catalog["year"].map(
        lambda value: str(int(value)) if pd.notna(value) else "unknown"
    )
    catalog["duplicate_key"] = [
        "|".join(
            (
                normalize_identifier(brand),
                normalize_identifier(name),
                year,
            )
        )
        for brand, name, year in zip(
            catalog["brand"], catalog["name"], year_key, strict=True
        )
    ]
    source_stats["matched_descriptions"] = int(
        catalog["source_description"].str.len().gt(0).sum()
    )
    return catalog, source_stats


def build_retrieval_text(row: pd.Series) -> str:
    """Build the future retrieval document without creating embeddings."""
    parts = [f"{row['name']} by {row['brand']}."]
    if pd.notna(row["gender"]):
        parts.append(f"Gender: {row['gender']}.")
    if row["accords"]:
        parts.append(f"Main accords: {', '.join(row['accords'])}.")
    parts.append(f"Top notes: {', '.join(row['top_notes'])}.")
    parts.append(f"Heart notes: {', '.join(row['middle_notes'])}.")
    parts.append(f"Base notes: {', '.join(row['base_notes'])}.")
    if row["source_description"]:
        parts.append(f"Description: {row['source_description']}")
    return " ".join(parts)
