"""Generate reproducible Russian training queries from perfume metadata."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl

NOTE_RU = {
    "musk": "мускус",
    "bergamot": "бергамот",
    "sandalwood": "сандал",
    "jasmine": "жасмин",
    "amber": "амбра",
    "patchouli": "пачули",
    "vanilla": "ваниль",
    "rose": "роза",
    "cedar": "кедр",
    "vetiver": "ветивер",
    "mandarin orange": "мандарин",
    "tonka bean": "бобы тонка",
    "lemon": "лимон",
    "orange blossom": "флердоранж",
    "pink pepper": "розовый перец",
    "lavender": "лаванда",
    "iris": "ирис",
    "leather": "кожа",
    "violet": "фиалка",
    "cardamom": "кардамон",
    "oakmoss": "дубовый мох",
    "grapefruit": "грейпфрут",
    "benzoin": "бензоин",
    "peach": "персик",
    "neroli": "нероли",
    "tuberose": "тубероза",
    "incense": "ладан",
    "pear": "груша",
    "ginger": "имбирь",
    "saffron": "шафран",
    "coconut": "кокос",
    "tobacco": "табак",
    "honey": "мёд",
    "cinnamon": "корица",
    "mint": "мята",
    "black currant": "чёрная смородина",
    "orange": "апельсин",
    "raspberry": "малина",
    "apple": "яблоко",
    "pineapple": "ананас",
}

ACCORD_RU = {
    "woody": "древесный",
    "citrus": "цитрусовый",
    "sweet": "сладкий",
    "aromatic": "ароматический",
    "fruity": "фруктовый",
    "powdery": "пудровый",
    "floral": "цветочный",
    "warm spicy": "тепло-пряный",
    "white floral": "белоцветочный",
    "vanilla": "ванильный",
    "fresh spicy": "свеже-пряный",
    "amber": "амбровый",
    "musky": "мускусный",
    "green": "зелёный",
    "fresh": "свежий",
    "rose": "розовый",
    "leather": "кожаный",
    "earthy": "землистый",
    "aquatic": "водный",
    "smoky": "дымный",
    "marine": "морской",
    "tobacco": "табачный",
}

TEMPLATES = {
    "note": [
        "Подбери парфюм с хорошо заметными нотами: {note1} и {note2}.",
        "Хочется композицию, где сочетаются {note1} и {note2}.",
        "Найди аромат с двумя основными нотами: {note1} и {note2}.",
    ],
    "accord": [
        "Подбери аромат с двумя направлениями: {accord1} и {accord2}.",
        "Ищу композицию, которая сочетает характеры: {accord1} и {accord2}.",
        "Хочется сочетание двух направлений в аромате: {accord1} и {accord2}.",
    ],
    "combo": [
        "Ищу {accord1} аромат с такими нотами: {note1} и {note2}.",
        "Подбери композицию: направление — {accord1}, главные ноты — {note1} и {note2}.",
        "Хочется {accord1} парфюм, в котором слышны {note1} и {note2}.",
    ],
    "negation": [
        "Ищу {accord1} аромат: нота {note1} нужна, а {excluded} нужно исключить.",
        "Подбери {accord1} композицию, где заметна нота {note1}, но исключи {excluded}.",
        "Хочется {accord1} парфюм: нота {note1} подходит, а {excluded} нужно исключить.",
    ],
}


def _terms(profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    notes = []
    for layer in ("top", "middle", "base"):
        for note in profile["notes"][layer]:
            key = str(note).casefold()
            if key in NOTE_RU and key not in notes:
                notes.append(key)
    accords = []
    for accord in profile["accords"]:
        key = str(accord).casefold()
        if key in ACCORD_RU and key not in accords:
            accords.append(key)
    return notes, accords


def _make_query(
    profile: dict[str, Any], variant: str, template_index: int
) -> tuple[str, dict[str, Any]] | None:
    notes, accords = _terms(profile)
    if len(notes) < 2 or len(accords) < 2:
        return None
    note1, note2 = notes[:2]
    accord1, accord2 = accords[:2]
    constraints = {
        "required_notes": [],
        "preferred_notes": [],
        "excluded_notes": [],
        "preferred_accords": [],
    }
    values = {
        "note1": NOTE_RU[note1],
        "note2": NOTE_RU[note2],
        "accord1": ACCORD_RU[accord1],
        "accord2": ACCORD_RU[accord2],
    }
    if variant == "note":
        constraints["required_notes"] = [note1]
        constraints["preferred_notes"] = [note2]
    elif variant == "accord":
        constraints["preferred_accords"] = [accord1, accord2]
    elif variant == "combo":
        constraints["required_notes"] = [note1]
        constraints["preferred_notes"] = [note2]
        constraints["preferred_accords"] = [accord1]
    else:
        excluded = next((key for key in NOTE_RU if key not in notes), None)
        if excluded is None:
            return None
        values["excluded"] = NOTE_RU[excluded]
        constraints["required_notes"] = [note1]
        constraints["excluded_notes"] = [excluded]
        constraints["preferred_accords"] = [accord1]
    template = TEMPLATES[variant][template_index % len(TEMPLATES[variant])]
    return template.format(**values), constraints


def generate_queries(
    profiles: list[dict[str, Any]],
    train_size: int = 2_000,
    validation_size: int = 200,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create unique queries with disjoint positive IDs across splits."""
    shuffled = profiles.copy()
    random.Random(seed).shuffle(shuffled)
    target = train_size + validation_size
    rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    variants = tuple(TEMPLATES)
    for profile in shuffled:
        index = len(rows)
        variant = variants[index % len(variants)]
        generated = _make_query(profile, variant, int(profile["perfume_id"]))
        if generated is None:
            continue
        text, constraints = generated
        text_key = text.casefold()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        split = "train" if len(rows) < train_size else "validation"
        prefix = "t" if split == "train" else "v"
        split_index = len(rows) + 1 if split == "train" else len(rows) - train_size + 1
        rows.append(
            {
                "query_id": f"{prefix}{split_index:05d}",
                "query_text": text,
                "split": split,
                "query_type": variant,
                "positive_perfume_id": str(profile["perfume_id"]),
                "constraints": constraints,
                "source": "synthetic_metadata_templates_v1",
            }
        )
        if len(rows) == target:
            break
    if len(rows) != target:
        raise RuntimeError(f"Generated only {len(rows)} unique queries; expected {target}")
    report = {
        "status": "passed",
        "seed": seed,
        "train_queries": train_size,
        "validation_queries": validation_size,
        "unique_positive_perfumes": len(
            {row["positive_perfume_id"] for row in rows}
        ),
        "query_type_distribution": dict(Counter(row["query_type"] for row in rows)),
        "eval_query_ids_used": 0,
        "warning": "Synthetic metadata queries are silver training data.",
    }
    return rows, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/processed/perfume_profiles_5000.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/training/synthetic_queries_v1.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/interim/training_queries_report.json")
    )
    parser.add_argument("--train-size", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, report = generate_queries(
        read_jsonl(args.profiles.resolve()),
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Synthetic training queries created.")
    print(f"Train: {report['train_queries']}")
    print(f"Validation: {report['validation_queries']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
