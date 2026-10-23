"""Deterministic bilingual query-intent parser for recommendation requests."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scentmatch.data.audit import read_jsonl
from scentmatch.retrieval.constraints import (
    ACCORD_ALIASES,
    NOTE_ALIASES,
    extract_exclusions,
    extract_preferences,
)
from scentmatch.retrieval.reference import (
    ReferenceResolution,
    extract_reference_text,
    has_budget_intent,
    resolve_reference_detailed,
)

Gender = Literal["any", "men", "women", "unisex"]
Level = Literal["any", "low", "high"]
Popularity = Literal["any", "high"]
Availability = Literal["any", "prefer_common"]
BudgetMode = Literal["any", "affordable", "cheaper_than_reference", "max_price"]

SEASON_ALIASES = {
    "spring": ("весна", "весен", "spring"),
    "summer": ("лето", "летн", "жар", "summer", "hot weather"),
    "autumn": ("осень", "осенн", "autumn", "fall"),
    "winter": ("зима", "зимн", "холод", "winter", "cold weather"),
}

OCCASION_ALIASES = {
    "office": ("офис", "работ", "делов", "office", "work"),
    "evening": ("вечер", "вечерин", "night out", "evening", "party"),
    "date": ("свидан", "романтич", "date", "romantic"),
    "everyday": ("каждый день", "повседнев", "daily", "everyday"),
    "sport": ("спорт", "трениров", "зал", "sport", "gym", "workout"),
    "vacation": ("отпуск", "пляж", "море", "vacation", "holiday", "beach"),
    "formal": ("торжеств", "официальн", "formal", "ceremony"),
}

LOW_SWEETNESS_PATTERNS = (
    r"\bнеслад\w*",
    r"\bне\s+слад\w*",
    r"\bбез\s+слад\w*",
    r"\bnot\s+(?:too\s+)?sweet\b",
    r"\bnon[-\s]?sweet\b",
    r"\bless\s+sweet\b",
)
HIGH_SWEETNESS_PATTERNS = (r"\bслад\w*", r"\bsweet\w*")
POPULARITY_PATTERNS = (
    r"\bлучш\w*",
    r"\bпопуляр\w*",
    r"\bизвестн\w*",
    r"\bбестселлер\w*",
    r"\bpopular\b",
    r"\bwell[-\s]?known\b",
    r"\bbestseller\b",
    r"\bmainstream\b",
    r"\bbest\b",
)
AVAILABILITY_PATTERNS = (
    r"\bлегко\s+найти\b",
    r"\bв\s+обычн\w*\s+магазин\w*",
    r"\bшироко\s+доступ\w*",
    r"\beasy\s+to\s+find\b",
    r"\bwidely\s+available\b",
    r"\bcommon\b",
)
HIGH_LONGEVITY_PATTERNS = (
    r"\bстойк\w*",
    r"\bдолго\s+держ\w*",
    r"\blong[-\s]?lasting\b",
    r"\bexcellent\s+longevity\b",
)
LOW_LONGEVITY_PATTERNS = (
    r"\bнестойк\w*",
    r"\bлегк\w*\s+стойк\w*",
    r"\bshort[-\s]?lived\b",
)
STRONG_SILLAGE_PATTERNS = (
    r"\bшлейф\w*",
    r"\bзаметн\w*\s+окружающ\w*",
    r"\bstrong\s+(?:sillage|projection)\b",
    r"\bproject\w*\b",
)
SOFT_SILLAGE_PATTERNS = (
    r"\bбез\s+сильн\w*\s+шлейф\w*",
    r"\bтих\w*\s+шлейф\w*",
    r"\bsoft\s+(?:sillage|projection)\b",
    r"\bskin\s+scent\b",
)

MAX_PRICE_PATTERNS = (
    re.compile(
        r"(?:до|не\s+дороже)\s*(?P<amount>\d[\d\s.,]*)\s*"
        r"(?P<currency>₽|руб(?:лей|ля|ль)?|rub)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:under|max(?:imum)?|up\s+to)\s*"
        r"(?P<currency>\$|usd|€|eur)?\s*(?P<amount>\d[\d\s.,]*)"
        r"\s*(?P<currency_tail>\$|usd|€|eur)?",
        re.IGNORECASE,
    ),
)


def _contains(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _aliases(text: str, mapping: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(
        canonical
        for canonical, aliases in mapping.items()
        if any(alias in text for alias in aliases)
    )


def _gender(text: str) -> Gender:
    if re.search(r"\b(?:унисекс|unisex)\b", text):
        return "unisex"
    if re.search(r"\b(?:женск\w*|для\s+женщин\w*|women'?s?|female)\b", text):
        return "women"
    if re.search(r"\b(?:мужск\w*|для\s+мужчин\w*|men'?s?|male)\b", text):
        return "men"
    return "any"


def _level(
    text: str, low_patterns: tuple[str, ...], high_patterns: tuple[str, ...]
) -> Level:
    if _contains(text, low_patterns):
        return "low"
    if _contains(text, high_patterns):
        return "high"
    return "any"


def _parse_max_price(text: str) -> tuple[float | None, str | None]:
    for pattern in MAX_PRICE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        amount_text = match.group("amount").replace(" ", "").replace(",", ".")
        try:
            amount = float(amount_text)
        except ValueError:
            continue
        currency_text = (
            match.groupdict().get("currency")
            or match.groupdict().get("currency_tail")
            or ""
        ).casefold()
        currency = None
        if currency_text in {"₽", "rub"} or currency_text.startswith("руб"):
            currency = "RUB"
        elif currency_text in {"$", "usd"}:
            currency = "USD"
        elif currency_text in {"€", "eur"}:
            currency = "EUR"
        return amount, currency
    return None, None


@dataclass(frozen=True)
class QueryIntent:
    query_text: str
    language: Literal["ru", "en"]
    gender: Gender
    seasons: tuple[str, ...]
    occasions: tuple[str, ...]
    preferred_notes: tuple[str, ...]
    excluded_notes: tuple[str, ...]
    preferred_accords: tuple[str, ...]
    excluded_accords: tuple[str, ...]
    sweetness: Level
    popularity: Popularity
    availability: Availability
    longevity: Level
    sillage: Literal["any", "soft", "strong"]
    budget_mode: BudgetMode
    max_price: float | None
    currency: str | None
    reference: ReferenceResolution | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "language": self.language,
            "gender": self.gender,
            "seasons": list(self.seasons),
            "occasions": list(self.occasions),
            "preferred_notes": list(self.preferred_notes),
            "excluded_notes": list(self.excluded_notes),
            "preferred_accords": list(self.preferred_accords),
            "excluded_accords": list(self.excluded_accords),
            "sweetness": self.sweetness,
            "popularity": self.popularity,
            "availability": self.availability,
            "longevity": self.longevity,
            "sillage": self.sillage,
            "budget_mode": self.budget_mode,
            "max_price": self.max_price,
            "currency": self.currency,
            "reference": (
                self.reference.to_dict()
                if self.reference is not None
                and (self.reference.raw_text is not None or self.reference.accepted)
                else None
            ),
        }

    @property
    def hard_constraints(self) -> tuple[str, ...]:
        constraints = []
        if self.gender != "any":
            constraints.append("gender")
        if self.excluded_notes or self.excluded_accords:
            constraints.append("explicit_exclusions")
        if self.sweetness == "low":
            constraints.append("low_sweetness")
        if self.max_price is not None:
            constraints.append("max_price")
        return tuple(constraints)


def parse_query_intent(
    query_text: str, profiles: list[dict[str, Any]] | None = None
) -> QueryIntent:
    """Parse explicit request attributes without an LLM or network call."""
    text = query_text.casefold().replace("ё", "е")
    exclusions = extract_exclusions(query_text)
    preferences = extract_preferences(query_text)
    preferred_notes = tuple(sorted(preferences & set(NOTE_ALIASES)))
    preferred_accords = tuple(
        sorted((preferences - set(NOTE_ALIASES)) | (preferences & set(ACCORD_ALIASES)))
    )
    sweetness = _level(text, LOW_SWEETNESS_PATTERNS, HIGH_SWEETNESS_PATTERNS)
    popularity: Popularity = (
        "high" if _contains(text, POPULARITY_PATTERNS) else "any"
    )
    availability: Availability = (
        "prefer_common"
        if popularity == "high" or _contains(text, AVAILABILITY_PATTERNS)
        else "any"
    )
    longevity = _level(text, LOW_LONGEVITY_PATTERNS, HIGH_LONGEVITY_PATTERNS)
    if _contains(text, SOFT_SILLAGE_PATTERNS):
        sillage: Literal["any", "soft", "strong"] = "soft"
    elif _contains(text, STRONG_SILLAGE_PATTERNS):
        sillage = "strong"
    else:
        sillage = "any"

    max_price, currency = _parse_max_price(text)
    reference_text = extract_reference_text(query_text)
    if max_price is not None:
        budget_mode: BudgetMode = "max_price"
    elif has_budget_intent(query_text):
        budget_mode = (
            "cheaper_than_reference" if reference_text is not None else "affordable"
        )
    else:
        budget_mode = "any"

    reference = (
        resolve_reference_detailed(query_text, profiles) if profiles is not None else None
    )
    return QueryIntent(
        query_text=query_text,
        language="ru" if re.search(r"[а-яё]", query_text, re.IGNORECASE) else "en",
        gender=_gender(text),
        seasons=tuple(sorted(_aliases(text, SEASON_ALIASES))),
        occasions=tuple(sorted(_aliases(text, OCCASION_ALIASES))),
        preferred_notes=preferred_notes,
        excluded_notes=tuple(sorted(exclusions["notes"])),
        preferred_accords=preferred_accords,
        excluded_accords=tuple(sorted(exclusions["accords"])),
        sweetness=sweetness,
        popularity=popularity,
        availability=availability,
        longevity=longevity,
        sillage=sillage,
        budget_mode=budget_mode,
        max_price=max_price,
        currency=currency,
        reference=reference,
    )


def gender_matches_intent(intent: QueryIntent, profile: dict[str, Any]) -> bool:
    """Apply an explicit query gender as a hard catalog constraint."""
    if intent.gender == "any":
        return True
    candidate_gender = str(profile.get("gender", "")).casefold()
    if intent.gender == "unisex":
        return candidate_gender == "unisex"
    return candidate_gender in {intent.gender, "unisex"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/processed/perfume_profiles_5000.jsonl"),
    )
    args = parser.parse_args()
    profiles = read_jsonl(args.profiles.resolve())
    intent = parse_query_intent(args.query, profiles)
    print(json.dumps(intent.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
