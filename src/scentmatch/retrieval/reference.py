"""Reference perfume resolution and lightweight recommendation constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from scentmatch.data.audit import read_jsonl

BUDGET_PATTERNS = (
    r"\bбюджет\w*",
    r"\bдешев\w*",
    r"\bнедорог\w*",
    r"\bдоступн\w*\s+по\s+цене",
    r"\bcheaper\b",
    r"\bbudget(?:-friendly)?\b",
    r"\bmore\s+affordable\b",
    r"\baffordable\b",
    r"\bless\s+expensive\b",
)

REFERENCE_CUE_RE = re.compile(
    r"(?:похож\w*\s+на|аналог\w*(?:\s+для|\s+аромата)?|"
    r"similar\s+to|something\s+like|alternative\s+to|inspired\s+by)\s+"
    r"(?P<reference>.+)",
    flags=re.IGNORECASE,
)

REFERENCE_TAIL_RE = re.compile(
    r"\s+(?:но|только|котор\w*|при\s+этом|без|but|without|with|that|which)\b.*$",
    flags=re.IGNORECASE,
)

GENERIC_NAME_TOKENS = {
    "a",
    "de",
    "eau",
    "edp",
    "edt",
    "for",
    "homme",
    "l",
    "la",
    "le",
    "of",
    "parfum",
    "perfume",
    "pour",
    "the",
    "toilette",
    "woman",
    "women",
}


@dataclass(frozen=True)
class ReferenceResolution:
    """Auditable outcome of catalog entity resolution."""

    raw_text: str | None
    profile: dict[str, Any] | None
    confidence: float
    margin: float
    method: str
    accepted: bool
    alternatives: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "resolved_id": (
                str(self.profile["perfume_id"]) if self.profile is not None else None
            ),
            "resolved_name": (
                f"{display_name(self.profile['name'])} — "
                f"{display_name(self.profile['brand'])}"
                if self.profile is not None
                else None
            ),
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "method": self.method,
            "accepted": self.accepted,
            "alternatives": list(self.alternatives),
        }


def has_budget_intent(query: str) -> bool:
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in BUDGET_PATTERNS)


def _normalise_name(value: str) -> str:
    text = value.casefold().replace("ё", "е").replace("&", " and ")
    text = re.sub(r"\bedp\b", " eau de parfum ", text)
    text = re.sub(r"\bedt\b", " eau de toilette ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _informative_tokens(value: str) -> list[str]:
    return [
        token
        for token in _normalise_name(value).split()
        if len(token) >= 3 and token not in GENERIC_NAME_TOKENS
    ]


def extract_reference_text(query: str) -> str | None:
    """Return the likely perfume-name span following an explicit comparison cue."""
    match = REFERENCE_CUE_RE.search(query)
    if match is None:
        return None
    fragment = REFERENCE_TAIL_RE.sub("", match.group("reference"))
    fragment = fragment.strip(" \t\r\n.,;:!?()[]{}\"'")
    return fragment or None


def _candidate_score(fragment: str, profile: dict[str, Any]) -> tuple[float, str]:
    name = _normalise_name(str(profile["name"]).replace("-", " "))
    brand = _normalise_name(str(profile["brand"]).replace("-", " "))
    full = f"{name} {brand}".strip()
    brand_first = f"{brand} {name}".strip()
    aliases = {name, full, brand_first}
    fragment_key = _normalise_name(fragment)
    # Dataset names sometimes append a launch year to the original edition.
    # A query without that year should still resolve to the original, not a flanker.
    name_without_year = re.sub(r"\s+(?:19|20)\d{2}$", "", name)
    if name_without_year != name and fragment_key in {
        name_without_year,
        f"{name_without_year} {brand}".strip(),
        f"{brand} {name_without_year}".strip(),
    }:
        return 99.0, "exact_original_without_year"

    name_tokens = _informative_tokens(name)
    fragment_tokens = _informative_tokens(fragment_key)
    if not name_tokens:
        # Names such as "Homme" are not unique enough without a matching brand.
        brand_similarity = fuzz.ratio(" ".join(fragment_tokens), brand)
        if brand_similarity >= 90.0:
            return 92.0, "brand_plus_generic_name"
        return min(58.0, float(brand_similarity)), "generic_name_rejected"

    if any(alias and alias == fragment_key for alias in aliases):
        return 100.0, "exact"

    alias_score = max(
        max(fuzz.ratio(fragment_key, alias), fuzz.WRatio(fragment_key, alias))
        for alias in aliases
        if alias
    )
    informative_score = fuzz.WRatio(
        " ".join(fragment_tokens),
        " ".join(dict.fromkeys(name_tokens + _informative_tokens(brand))),
    )
    score = 0.78 * float(alias_score) + 0.22 * float(informative_score)
    return min(99.5, score), "fuzzy"


def resolve_reference_detailed(
    query: str,
    profiles: list[dict[str, Any]],
    *,
    min_confidence: float = 0.78,
    min_margin: float = 0.025,
) -> ReferenceResolution:
    """Resolve a catalog perfume and retain confidence, margin and alternatives."""
    fragment = extract_reference_text(query)
    if fragment is None:
        # Preserve support for exact catalog mentions without a comparison cue.
        query_key = _normalise_name(query)
        exact_matches: list[dict[str, Any]] = []
        for profile in profiles:
            name = _normalise_name(str(profile["name"]).replace("-", " "))
            brand = _normalise_name(str(profile["brand"]).replace("-", " "))
            if len(name) >= 5 and (
                f"{name} {brand}" in query_key or f"{brand} {name}" in query_key
            ):
                exact_matches.append(profile)
        if not exact_matches:
            return ReferenceResolution(None, None, 0.0, 0.0, "no_reference_cue", False)
        exact_matches.sort(key=lambda item: len(str(item["name"])), reverse=True)
        profile = exact_matches[0]
        return ReferenceResolution(
            f"{profile['name']} {profile['brand']}",
            profile,
            1.0,
            1.0,
            "exact_without_cue",
            True,
        )

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for profile in profiles:
        score, method = _candidate_score(fragment, profile)
        scored.append((score, method, profile))
    scored.sort(
        key=lambda item: (item[0], len(str(item[2]["name"]))), reverse=True
    )
    if not scored:
        return ReferenceResolution(fragment, None, 0.0, 0.0, "empty_catalog", False)

    best_score, method, best_profile = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    confidence = best_score / 100.0
    margin = max(0.0, (best_score - second_score) / 100.0)
    accepted = method == "exact" or (
        confidence >= min_confidence and margin >= min_margin
    )
    alternatives = tuple(
        {
            "perfume_id": str(profile["perfume_id"]),
            "name": display_name(str(profile["name"])),
            "brand": display_name(str(profile["brand"])),
            "score": round(score / 100.0, 4),
        }
        for score, _, profile in scored[:3]
    )
    return ReferenceResolution(
        raw_text=fragment,
        profile=best_profile if accepted else None,
        confidence=confidence,
        margin=margin,
        method=method if accepted else "low_confidence",
        accepted=accepted,
        alternatives=alternatives,
    )


def resolve_reference(
    query: str, profiles: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Backward-compatible wrapper returning only an accepted catalog profile."""
    return resolve_reference_detailed(query, profiles).profile


def load_optional_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["perfume_id"]): row for row in read_jsonl(path)}


def display_name(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("-", " ").split())


def gender_compatible(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    reference_gender = str(reference.get("gender", "")).casefold()
    candidate_gender = str(candidate.get("gender", "")).casefold()
    if reference_gender not in {"men", "women"}:
        return True
    return candidate_gender in {reference_gender, "unisex"}
