"""Deterministic, evidence-backed explanations for perfume recommendations."""

from __future__ import annotations

from typing import Any

SEASONS = {
    "ru": {
        "spring": "весны",
        "summer": "лета",
        "autumn": "осени",
        "winter": "зимы",
    },
    "en": {
        "spring": "spring",
        "summer": "summer",
        "autumn": "autumn",
        "winter": "winter",
    },
}

OCCASIONS = {
    "ru": {
        "office": "офиса",
        "evening": "вечера",
        "date": "свидания",
        "everyday": "повседневного использования",
        "sport": "спорта",
        "vacation": "отпуска",
        "formal": "формального случая",
    },
    "en": {
        "office": "office wear",
        "evening": "evening wear",
        "date": "a date",
        "everyday": "everyday wear",
        "sport": "sport",
        "vacation": "a vacation",
        "formal": "a formal occasion",
    },
}


def _strength(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _joined(values: list[str], language: str) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    separator = " и " if language == "ru" else " and "
    return ", ".join(values[:-1]) + separator + values[-1]


def summarize_intent(intent: dict[str, Any]) -> list[str]:
    """Return compact, user-facing labels for parsed preferences."""
    language = "ru" if intent.get("language") == "ru" else "en"
    labels: list[str] = []
    gender = intent.get("gender", "any")
    if gender != "any":
        gender_labels = {
            "ru": {"men": "мужской", "women": "женский", "unisex": "унисекс"},
            "en": {"men": "men", "women": "women", "unisex": "unisex"},
        }
        labels.append(gender_labels[language].get(gender, str(gender)))
    for season in intent.get("seasons", []):
        labels.append(SEASONS[language].get(season, str(season)))
    for occasion in intent.get("occasions", []):
        labels.append(OCCASIONS[language].get(occasion, str(occasion)))
    level_labels = {
        "ru": {
            ("sweetness", "low"): "несладкий",
            ("sweetness", "high"): "сладкий",
            ("longevity", "high"): "стойкий",
            ("longevity", "low"): "с невысокой стойкостью",
            ("sillage", "strong"): "с сильным шлейфом",
            ("sillage", "soft"): "с мягким шлейфом",
            ("popularity", "high"): "популярный",
        },
        "en": {
            ("sweetness", "low"): "not sweet",
            ("sweetness", "high"): "sweet",
            ("longevity", "high"): "long-lasting",
            ("longevity", "low"): "shorter-lasting",
            ("sillage", "strong"): "strong sillage",
            ("sillage", "soft"): "soft sillage",
            ("popularity", "high"): "popular",
        },
    }
    for field in ("sweetness", "longevity", "sillage", "popularity"):
        value = intent.get(field, "any")
        label = level_labels[language].get((field, value))
        if label:
            labels.append(label)
    budget_mode = intent.get("budget_mode", "any")
    if budget_mode != "any":
        budget_labels = {
            "ru": {
                "affordable": "недорогой",
                "cheaper_than_reference": "дешевле референса",
                "max_price": "с ограничением цены",
            },
            "en": {
                "affordable": "affordable",
                "cheaper_than_reference": "cheaper than the reference",
                "max_price": "price-limited",
            },
        }
        labels.append(budget_labels[language].get(budget_mode, str(budget_mode)))
    return labels


class ExplanationGenerator:
    """Explain only signals that were available to the ranker."""

    def explain(
        self,
        intent: dict[str, Any],
        result: dict[str, Any],
        reference_name: str | None = None,
    ) -> dict[str, Any]:
        language = "ru" if intent.get("language") == "ru" else "en"
        components = result.get("components", {})
        reasons: list[dict[str, Any]] = []

        def add(reason_type: str, score: float | None, ru: str, en: str) -> None:
            if score is None:
                return
            reasons.append(
                {
                    "type": reason_type,
                    "text": ru if language == "ru" else en,
                    "strength": _strength(float(score)),
                    "score": round(float(score), 4),
                }
            )

        reminds = components.get("reminds_me_of")
        if reference_name and reminds is not None and float(reminds) > 0.05:
            add(
                "community_similarity",
                float(reminds),
                f"Пользователи сравнивают его с {reference_name}",
                f"Community members compare it with {reference_name}",
            )

        link_score = components.get("link_prediction")
        if reference_name and link_score is not None and float(link_score) >= 0.55:
            add(
                "ml_similarity",
                float(link_score),
                "ML-модель сходства поддерживает эту пару ароматов",
                "The ML similarity model supports this perfume pair",
            )

        reference_fit = components.get("reference_content")
        if (
            reference_name
            and reference_fit is not None
            and float(reference_fit) >= 0.5
        ):
            add(
                "profile_similarity",
                float(reference_fit),
                f"Профиль нот и аккордов близок к {reference_name}",
                f"Its notes and accords are close to {reference_name}",
            )

        seasons = list(intent.get("seasons", []))
        season_fit = components.get("season_fit")
        if seasons and season_fit is not None:
            season_text = _joined(
                [SEASONS[language].get(item, str(item)) for item in seasons],
                language,
            )
            add(
                "season_fit",
                float(season_fit),
                f"Community-голоса поддерживают использование для {season_text}",
                f"Community votes support wearing it in {season_text}",
            )

        occasions = list(intent.get("occasions", []))
        daypart_fit = components.get("daypart_fit")
        if occasions and daypart_fit is not None:
            occasion_text = _joined(
                [OCCASIONS[language].get(item, str(item)) for item in occasions],
                language,
            )
            add(
                "occasion_fit",
                float(daypart_fit),
                f"Режим day/night соответствует запросу для {occasion_text}",
                f"Its day/night profile matches {occasion_text}",
            )

        longevity_fit = components.get("longevity_fit")
        if intent.get("longevity") != "any" and longevity_fit is not None:
            high = intent.get("longevity") == "high"
            add(
                "longevity_fit",
                float(longevity_fit),
                "Стойкость соответствует запросу"
                if high
                else "Более лёгкая стойкость соответствует запросу",
                "Its longevity matches the request"
                if high
                else "Its lighter longevity matches the request",
            )

        sillage_fit = components.get("sillage_fit")
        if intent.get("sillage") != "any" and sillage_fit is not None:
            soft = intent.get("sillage") == "soft"
            add(
                "sillage_fit",
                float(sillage_fit),
                "Шлейф достаточно мягкий для запроса"
                if soft
                else "Выраженность шлейфа соответствует запросу",
                "Its sillage is soft enough for the request"
                if soft
                else "Its sillage strength matches the request",
            )

        sweetness_fit = components.get("sweetness_fit")
        if intent.get("sweetness") != "any" and sweetness_fit is not None:
            low = intent.get("sweetness") == "low"
            add(
                "sweetness_fit",
                float(sweetness_fit),
                "Сладкий аккорд не доминирует"
                if low
                else "Выраженная сладость соответствует запросу",
                "The sweet accord is not dominant"
                if low
                else "Its pronounced sweetness matches the request",
            )

        popularity = components.get("popularity")
        if intent.get("popularity") == "high" and popularity is not None:
            add(
                "popularity",
                float(popularity),
                "Аромат находится в популярной части каталога",
                "The perfume is in the popular part of the catalogue",
            )

        budget_fit = components.get("budget_fit")
        if intent.get("budget_mode") != "any" and budget_fit is not None:
            add(
                "budget_fit",
                float(budget_fit),
                "Исторический ценовой сигнал или каталожный сегмент "
                "поддерживает бюджетное ограничение",
                "A historical price signal or catalogue segment supports "
                "the budget constraint",
            )

        preferences = list(intent.get("preferred_notes", [])) + list(
            intent.get("preferred_accords", [])
        )
        structured_fit = components.get("structured_fit")
        if preferences and structured_fit is not None and float(structured_fit) > 0:
            preference_text = _joined([str(item) for item in preferences], language)
            add(
                "preferred_profile",
                float(structured_fit),
                f"В профиле присутствуют предпочтения: {preference_text}",
                f"Its profile contains the requested preferences: {preference_text}",
            )

        semantic = components.get("semantic_query")
        if len(reasons) < 3 and semantic is not None:
            add(
                "semantic_match",
                float(semantic),
                "Текстовое описание аромата соответствует смыслу запроса",
                "The perfume description semantically matches the request",
            )

        reasons = reasons[:4]
        drawbacks = []
        for item in result.get("community_evidence", {}).get("cons", [])[:2]:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            drawbacks.append(
                {
                    "type": "community_cons",
                    "text": text,
                    "agreement": round(float(item.get("agreement", 0.0)), 4),
                    "votes": int(item.get("up_votes", 0))
                    + int(item.get("down_votes", 0)),
                    "source_language": "en",
                }
            )

        direct_support = float(components.get("reminds_me_of") or 0.0)
        ml_support = float(components.get("link_prediction") or 0.0)
        profile_support = float(components.get("reference_content") or 0.0)
        if direct_support >= 0.35 or (ml_support >= 0.75 and profile_support >= 0.65):
            evidence_level = "high"
        elif len(reasons) >= 2:
            evidence_level = "medium"
        else:
            evidence_level = "limited"

        if reasons:
            short = " ".join(reason["text"] + "." for reason in reasons[:2])
        else:
            short = (
                "Подтверждённых оснований пока недостаточно."
                if language == "ru"
                else "There is not enough confirmed evidence yet."
            )
        return {
            "short_explanation": short,
            "why_recommended": reasons,
            "possible_drawbacks": drawbacks,
            "evidence_level": evidence_level,
        }
