"""Small deterministic parser for explicit Russian and English exclusions."""

from __future__ import annotations

import re
from typing import Any

NOTE_ALIASES = {
    "vanilla": ("ваниль", "ванили", "vanilla"),
    "rose": ("роза", "розы", "розой", "rose"),
    "patchouli": ("пачули", "patchouli"),
    "leather": ("кожа", "кожи", "кожей", "leather"),
    "tobacco": ("табак", "табака", "tobacco"),
    "cinnamon": ("корица", "корицы", "cinnamon"),
    "coconut": ("кокос", "кокоса", "кокосом", "coconut"),
    "incense": ("ладан", "ладана", "incense"),
    "caramel": ("карамель", "карамельной", "caramel"),
    "bergamot": ("бергамот", "bergamot"),
    "neroli": ("нероли", "neroli"),
    "iris": ("ирис", "iris"),
    "cedar": ("кедр", "cedar"),
    "vetiver": ("ветивер", "vetiver"),
    "lavender": ("лаванда", "лаванды", "lavender"),
    "cardamom": ("кардамон", "cardamom"),
    "musk": ("мускус", "мускуса", "musk"),
    "lemon": ("лимон", "лимона", "lemon"),
    "orange": ("апельсин", "апельсина", "orange"),
    "jasmine": ("жасмин", "жасмина", "jasmine"),
    "amber": ("амбра", "амбры", "амброй", "amber"),
    "oud": ("уд", "уда", "oud"),
    "pineapple": ("ананас", "ананаса", "pineapple"),
    "apple": ("яблоко", "яблока", "apple"),
}

ACCORD_ALIASES = {
    "sweet": ("сладость", "сладости", "сладкого", "сладкий", "sweet"),
    "fruity": ("фруктовый", "фруктового", "фруктовости", "fruity"),
    "smoky": ("дымный", "дымного", "дымность", "дымности", "smoky"),
    "tropical": ("тропический", "тропического", "tropical"),
    "leather": ("кожа", "кожи", "кожей", "кожаного", "leather"),
    "tobacco": ("табак", "табака", "табачного", "tobacco"),
    "caramel": ("карамель", "карамельной", "caramel"),
    "powdery": ("пудровый", "пудрового", "пудровости", "powdery"),
    "aquatic": ("акватический", "водный", "aquatic"),
    "fresh": ("свежий", "свежего", "fresh"),
    "warm spicy": ("пряный", "тепло-пряный", "warm spicy"),
}

PREFERENCE_ALIASES = {
    "smoky": ("дымный", "дымного", "smoky"),
    "woody": ("древесный", "древесного", "woody"),
    "citrus": ("цитрусовый", "цитрусового", "citrus"),
    "powdery": ("пудровый", "пудрового", "powdery"),
    "aromatic": ("ароматический", "aromatic"),
    "leather": ("кожаный", "кожаного", "кожа", "кожей", "leather"),
    "tobacco": ("табачный", "табачного", "tobacco"),
    "coconut": ("кокос", "кокоса", "кокосом", "coconut"),
    "incense": ("ладан", "ладана", "incense"),
    "vanilla": ("ваниль", "ванили", "vanilla"),
    "bergamot": ("бергамот", "bergamot"),
    "neroli": ("нероли", "neroli"),
    "rose": ("роза", "розы", "розой", "rose"),
    "patchouli": ("пачули", "patchouli"),
    "iris": ("ирис", "iris"),
    "cedar": ("кедр", "cedar"),
    "vetiver": ("ветивер", "vetiver"),
    "lavender": ("лаванда", "лавандой", "lavender"),
    "cardamom": ("кардамон", "cardamom"),
    "musk": ("мускус", "мускусный", "musk"),
    "lemon": ("лимон", "лимона", "lemon"),
    "orange": ("апельсин", "апельсина", "orange"),
    "jasmine": ("жасмин", "жасмина", "jasmine"),
    "amber": ("амбра", "амбры", "амброй", "amber"),
    "oud": ("уд", "уда", "oud"),
    "pineapple": ("ананас", "ананаса", "pineapple"),
    "apple": ("яблоко", "яблока", "apple"),
    "aquatic": ("акватический", "водный", "aquatic"),
    "fresh": ("свежий", "свежего", "fresh"),
    "warm spicy": ("пряный", "тепло-пряный", "warm spicy"),
}


def extract_exclusions(query_text: str) -> dict[str, set[str]]:
    """Extract only explicitly negated notes and accords."""
    text = query_text.casefold().replace("ё", "е")
    segments = re.findall(r"\bбез\b([^.!?;]*)", text)
    segments.extend(re.findall(r"\bисключ\w*\b([^.!?;]*)", text))
    segments.extend(re.findall(r"\bwithout\b([^.!?;]*)", text))
    segments.extend(re.findall(r"\bno\b([^.!?;]*)", text))
    negative_context = " ".join(segments)

    def matched(mapping: dict[str, tuple[str, ...]]) -> set[str]:
        return {
            canonical
            for canonical, aliases in mapping.items()
            if any(alias in negative_context for alias in aliases)
        }

    return {
        "notes": matched(NOTE_ALIASES),
        "accords": matched(ACCORD_ALIASES),
    }


def extract_preferences(query_text: str) -> set[str]:
    """Extract explicit positive note/accord mentions in Russian or English."""
    text = query_text.casefold().replace("ё", "е")
    positive_context = re.sub(
        r"\b(?:без|исключ\w*|without|no)\b[^.!?;]*",
        " ",
        text,
    )
    return {
        canonical
        for canonical, aliases in PREFERENCE_ALIASES.items()
        if any(alias in positive_context for alias in aliases)
    }


def preference_bonus(profile: dict[str, Any], preferences: set[str]) -> float:
    notes = {
        str(note).casefold()
        for values in profile["notes"].values()
        for note in values
    }
    accords = {str(value).casefold() for value in profile["accords"]}
    return min(0.12, 0.04 * len(preferences & (notes | accords)))


def violates_exclusions(
    profile: dict[str, Any], exclusions: dict[str, set[str]]
) -> bool:
    notes = {
        str(note).casefold()
        for values in profile["notes"].values()
        for note in values
    }
    accords = {str(value).casefold() for value in profile["accords"]}
    return bool(
        (notes | accords) & exclusions["notes"]
        or accords & exclusions["accords"]
    )
