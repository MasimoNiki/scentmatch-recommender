"""Audit generated perfume profiles before retrieval experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from scentmatch.data.fragrantica import normalize_identifier

REQUIRED_FIELDS = {
    "schema_version",
    "perfume_id",
    "name",
    "brand",
    "source_url",
    "gender",
    "country",
    "year",
    "rating",
    "perfumers",
    "accords",
    "notes",
    "source_description",
    "retrieval_text",
}
PLACEHOLDERS = {"unknown", "none", "null", "nan", "n/a", "-"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and report the physical line for malformed JSON."""
    profiles = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"Profile on line {line_number} is not an object")
            profiles.append(value)
    return profiles


def _quantiles(values: Iterable[int]) -> dict[str, float]:
    series = pd.Series(list(values), dtype="int64")
    result = series.quantile([0, 0.5, 0.9, 0.95, 0.99, 1])
    return {str(key): round(float(value), 2) for key, value in result.items()}


def _sample(values: Iterable[str], limit: int = 20) -> list[str]:
    return sorted(set(values))[:limit]


def audit_profiles(
    profiles: list[dict[str, Any]], expected_count: int = 5_000
) -> dict[str, Any]:
    """Return machine-readable quality checks without changing source values."""
    errors: dict[str, list[str]] = defaultdict(list)
    warnings: dict[str, list[str]] = defaultdict(list)
    ids: list[str] = []
    duplicate_keys: list[str] = []
    retrieval_texts: list[str] = []
    description_lengths: list[int] = []
    retrieval_lengths: list[int] = []
    term_variants: dict[str, dict[str, set[str]]] = {
        "notes": defaultdict(set),
        "accords": defaultdict(set),
        "perfumers": defaultdict(set),
    }
    placeholder_ids: list[str] = []
    markup_ids: list[str] = []
    url_in_description_ids: list[str] = []
    replacement_character_ids: list[str] = []

    if len(profiles) != expected_count:
        errors["unexpected_profile_count"].append(
            f"expected {expected_count}, found {len(profiles)}"
        )

    for position, profile in enumerate(profiles, start=1):
        profile_id = str(profile.get("perfume_id", f"line:{position}"))
        ids.append(profile_id)
        missing = REQUIRED_FIELDS - set(profile)
        if missing:
            errors["missing_required_fields"].append(
                f"{profile_id}: {', '.join(sorted(missing))}"
            )
            continue
        if not profile_id.isdigit():
            errors["invalid_perfume_id"].append(profile_id)

        name = str(profile["name"]).strip()
        brand = str(profile["brand"]).strip()
        duplicate_keys.append(
            f"{normalize_identifier(brand)}|{normalize_identifier(name)}"
        )
        notes = profile.get("notes")
        if not isinstance(notes, dict):
            errors["invalid_notes_object"].append(profile_id)
            notes = {}
        for layer in ("top", "middle", "base"):
            terms = notes.get(layer, [])
            if not isinstance(terms, list) or not terms:
                errors[f"empty_or_invalid_{layer}_notes"].append(profile_id)
                continue
            if len(terms) != len(set(terms)):
                warnings[f"duplicate_{layer}_notes"].append(profile_id)
            for term in terms:
                original = str(term)
                term_variants["notes"][original.casefold()].add(original)
                if original.strip().casefold() in PLACEHOLDERS:
                    placeholder_ids.append(profile_id)

        for field in ("accords", "perfumers"):
            terms = profile.get(field, [])
            if not isinstance(terms, list):
                errors[f"invalid_{field}"].append(profile_id)
                continue
            if field == "accords" and not terms:
                errors["empty_accords"].append(profile_id)
            if len(terms) != len(set(terms)):
                warnings[f"duplicate_{field}"].append(profile_id)
            for term in terms:
                original = str(term)
                term_variants[field][original.casefold()].add(original)
                if original.strip().casefold() in PLACEHOLDERS:
                    placeholder_ids.append(profile_id)

        description = str(profile["source_description"])
        retrieval_text = str(profile["retrieval_text"])
        description_lengths.append(len(description))
        retrieval_lengths.append(len(retrieval_text))
        retrieval_texts.append(retrieval_text)
        if len(description) < 80:
            errors["short_description"].append(profile_id)
        if not retrieval_text.strip():
            errors["empty_retrieval_text"].append(profile_id)
        if re.search(r"<[^>]+>", description):
            markup_ids.append(profile_id)
        if re.search(r"https?://|www\.", description, flags=re.IGNORECASE):
            url_in_description_ids.append(profile_id)
        if "\ufffd" in description or "\ufffd" in retrieval_text:
            replacement_character_ids.append(profile_id)

    id_counts = Counter(ids)
    duplicate_id_values = [key for key, count in id_counts.items() if count > 1]
    if duplicate_id_values:
        errors["duplicate_perfume_ids"].extend(duplicate_id_values)
    text_counts = Counter(retrieval_texts)
    duplicate_text_count = sum(count - 1 for count in text_counts.values() if count > 1)
    if duplicate_text_count:
        errors["duplicate_retrieval_texts"].append(str(duplicate_text_count))

    name_key_counts = Counter(duplicate_keys)
    duplicate_name_keys = [
        key for key, count in name_key_counts.items() if count > 1
    ]
    case_variant_groups = {
        field: {
            key: sorted(values)
            for key, values in groups.items()
            if len(values) > 1
        }
        for field, groups in term_variants.items()
    }
    checks = {
        "errors": {key: _sample(values) for key, values in sorted(errors.items())},
        "warnings": {
            key: _sample(values) for key, values in sorted(warnings.items())
        },
    }
    error_occurrences = sum(len(values) for values in errors.values())
    return {
        "status": "passed" if error_occurrences == 0 else "failed",
        "profile_count": len(profiles),
        "unique_perfume_ids": len(id_counts),
        "checks": checks,
        "observations": {
            "brand_name_keys_repeated_across_editions": len(duplicate_name_keys),
            "brand_name_key_examples": _sample(duplicate_name_keys, 10),
            "description_contains_markup": len(set(markup_ids)),
            "description_contains_url": len(set(url_in_description_ids)),
            "replacement_character_rows": len(set(replacement_character_ids)),
            "placeholder_term_rows": len(set(placeholder_ids)),
            "case_variant_groups": case_variant_groups,
        },
        "lengths": {
            "source_description_characters": _quantiles(description_lengths),
            "retrieval_text_characters": _quantiles(retrieval_lengths),
        },
        "normalization_decision": {
            "source_fields": "preserve",
            "case_only_variants": "normalize only in future derived ML fields",
            "brand_name_editions": "preserve; same name can represent another year",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    observations = report["observations"]
    lengths = report["lengths"]
    errors = report["checks"]["errors"]
    warnings = report["checks"]["warnings"]
    error_count = sum(len(values) for values in errors.values())
    warning_count = sum(len(values) for values in warnings.values())
    return f"""# Аудит perfume profiles

Статус: **{report['status']}**.

- Профилей: {report['profile_count']}.
- Уникальных `perfume_id`: {report['unique_perfume_ids']}.
- Категорий ошибок: {len(errors)}; показанных примеров: {error_count}.
- Категорий предупреждений: {len(warnings)}; показанных примеров: {warning_count}.
- Повторяющихся `brand + name` между разными изданиями:
  {observations['brand_name_keys_repeated_across_editions']}.
- Описаний с HTML-подобной разметкой:
  {observations['description_contains_markup']}.
- Описаний с URL: {observations['description_contains_url']}.
- Строк с символом повреждённой кодировки:
  {observations['replacement_character_rows']}.
- Строк с placeholder-терминами: {observations['placeholder_term_rows']}.

## Длины текстов

- Медиана исходного описания:
  {lengths['source_description_characters']['0.5']} символов.
- 99-й процентиль `retrieval_text`:
  {lengths['retrieval_text_characters']['0.99']} символов.
- Максимальный `retrieval_text`:
  {lengths['retrieval_text_characters']['1.0']} символов.

## Решение по нормализации

Исходные названия нот, аккордов и парфюмеров сохраняем без переписывания.
Нормализованные варианты будем создавать позднее как производные ML-поля.
Одинаковые `brand + name` разных лет автоматически не объединяем: это могут
быть переиздания с изменённым составом.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/processed/perfume_profiles_5000.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--expected-count", type=int, default=5_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profiles = read_jsonl(args.profiles.resolve())
    report = audit_profiles(profiles, expected_count=args.expected_count)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "profile_audit.json"
    markdown_path = output_dir / "profile_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Profile audit {report['status']}.")
    print(f"Profiles: {report['profile_count']}")
    print(f"Unique IDs: {report['unique_perfume_ids']}")
    print(f"Report: {json_path}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
