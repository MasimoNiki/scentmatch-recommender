"""Run reproducible exploratory analysis of the selected perfume dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scentmatch.data.fragrantica import build_catalog


def top_counts(series: pd.Series, limit: int = 15) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).head(limit).items()
    }


def list_vocabulary(series: pd.Series) -> pd.Series:
    return series.explode().dropna().astype(str).str.strip().str.casefold()


def quantiles(series: pd.Series) -> dict[str, float]:
    values = series.dropna().astype(float).quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    return {str(key): round(float(value), 3) for key, value in values.items()}


def build_eda_summary(raw_dir: Path) -> dict[str, object]:
    catalog, source = build_catalog(raw_dir)
    full_pyramid = catalog[
        ["top_notes", "middle_notes", "base_notes"]
    ].apply(lambda column: column.map(bool)).all(axis=1)
    duplicate_rows = int(catalog["duplicate_key"].duplicated().sum())
    deduplicated = catalog.sort_values(
        ["rating_count", "description_length", "perfume_id"],
        ascending=[False, False, True],
    ).drop_duplicates("duplicate_key", keep="first")

    summary: dict[str, object] = {
        "source": source,
        "catalog": {
            "joined_rows": len(catalog),
            "deduplicated_rows": len(deduplicated),
            "duplicate_brand_name_year_rows": duplicate_rows,
            "unique_brands": int(catalog["brand"].nunique()),
            "unique_countries": int(catalog["country"].nunique()),
            "unique_primary_accords": int(catalog["primary_accord"].nunique()),
        },
        "completeness": {
            "full_pyramid_rows": int(full_pyramid.sum()),
            "full_pyramid_share": round(float(full_pyramid.mean()), 6),
            "description_rows": int(catalog["description_length"].gt(0).sum()),
            "description_share": round(
                float(catalog["description_length"].gt(0).mean()), 6
            ),
            "year_rows": int(catalog["year"].notna().sum()),
            "year_share": round(float(catalog["year"].notna().mean()), 6),
            "known_perfumer_rows": int(catalog["perfumers"].map(bool).sum()),
            "known_perfumer_share": round(
                float(catalog["perfumers"].map(bool).mean()), 6
            ),
        },
        "numeric_distributions": {
            "rating_count_quantiles": quantiles(catalog["rating_count"]),
            "rating_value_quantiles": quantiles(catalog["rating_value"]),
            "description_length_quantiles": quantiles(
                catalog["description_length"]
            ),
            "note_count_quantiles": quantiles(catalog["note_count"]),
            "year_min": int(catalog["year"].min()),
            "year_max": int(catalog["year"].max()),
        },
        "distributions": {
            "gender": top_counts(catalog["gender"]),
            "primary_accord": top_counts(catalog["primary_accord"], 25),
            "brand": top_counts(catalog["brand"], 20),
            "country": top_counts(catalog["country"], 20),
        },
        "note_vocabulary": {},
        "selection_candidate_counts": {},
    }
    note_summary: dict[str, object] = {}
    for layer in ("top_notes", "middle_notes", "base_notes"):
        vocabulary = list_vocabulary(catalog[layer])
        note_summary[layer] = {
            "unique": int(vocabulary.nunique()),
            "most_common": top_counts(vocabulary, 15),
        }
    summary["note_vocabulary"] = note_summary

    candidate_counts: dict[str, int] = {}
    for threshold in (50, 100, 200, 400):
        eligible = (
            deduplicated["rating_count"].ge(threshold)
            & deduplicated["description_length"].ge(80)
            & deduplicated[["top_notes", "middle_notes", "base_notes"]]
            .apply(lambda column: column.map(bool))
            .all(axis=1)
        )
        candidate_counts[f"rating_count_gte_{threshold}"] = int(eligible.sum())
    summary["selection_candidate_counts"] = candidate_counts
    return summary


def render_markdown(summary: dict[str, object]) -> str:
    source = summary["source"]
    catalog = summary["catalog"]
    completeness = summary["completeness"]
    numeric = summary["numeric_distributions"]
    distributions = summary["distributions"]
    candidates = summary["selection_candidate_counts"]

    def table(mapping: dict[str, int]) -> str:
        rows = ["| Значение | Записей |", "|---|---:|"]
        rows.extend(f"| {key} | {value} |" for key, value in mapping.items())
        return "\n".join(rows)

    return f"""# ScentMatch: первичный EDA

Отчёт создаётся командой `scentmatch-eda` из исходных CSV и не содержит
embeddings или результатов обучения.

## Размер и связывание

- Структурированный файл: {source['clean_rows']} строк.
- Raw-файл: {source['raw_rows']} строк.
- Повторяющихся ID в raw-файле: {source['raw_duplicate_ids']}.
- Профилей, связанных с описанием: {source['matched_descriptions']}.
- После дедупликации `brand + name + year`: {catalog['deduplicated_rows']}.
- Удаляемых повторов по этому ключу: {catalog['duplicate_brand_name_year_rows']}.

## Заполненность

- Полная пирамида нот: {completeness['full_pyramid_rows']}
  ({completeness['full_pyramid_share']:.1%}).
- Описание: {completeness['description_rows']}
  ({completeness['description_share']:.1%}).
- Год: {completeness['year_rows']} ({completeness['year_share']:.1%}).
- Известный парфюмер: {completeness['known_perfumer_rows']}
  ({completeness['known_perfumer_share']:.1%}).

## Основные распределения

### Гендер

{table(distributions['gender'])}

### Основной аккорд — первые 25

{table(distributions['primary_accord'])}

### Бренды — первые 20

{table(distributions['brand'])}

## Числовые признаки

- Число оценок: медиана {numeric['rating_count_quantiles']['0.5']},
  90-й процентиль {numeric['rating_count_quantiles']['0.9']}, максимум
  {numeric['rating_count_quantiles']['1.0']}.
- Рейтинг: медиана {numeric['rating_value_quantiles']['0.5']}.
- Годы: {numeric['year_min']}–{numeric['year_max']}.
- Длина описания: медиана
  {numeric['description_length_quantiles']['0.5']} символов.

## Размер пула при разных порогах

{table(candidates)}

## Выводы для отбора

1. Полнота пирамиды позволяет не восстанавливать слои эвристиками.
2. Raw-дубликаты нужно схлопывать по числовому ID из URL, выбирая самое длинное
   описание.
3. Повторы `brand + name + year` нужно дедуплицировать, оставляя запись с
   большим числом оценок.
4. Порог `rating_count >= 100` оставляет достаточный пул для выборки 5 000.
5. Нужен лимит на один бренд: крупнейшие бренды представлены сотнями записей.
6. Редкие основные аккорды следует объединять только для стратификации, сохраняя
   исходное значение в perfume profile.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/interim")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_eda_summary(args.raw_dir.resolve())
    json_path = output_dir / "eda_summary.json"
    markdown_path = output_dir / "eda_report.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print("EDA completed.")
    print(f"JSON summary: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
