from scentmatch.retrieval.explanations import ExplanationGenerator, summarize_intent


def _intent() -> dict:
    return {
        "language": "ru",
        "gender": "men",
        "seasons": ["summer"],
        "occasions": [],
        "preferred_notes": [],
        "excluded_notes": [],
        "preferred_accords": [],
        "excluded_accords": [],
        "sweetness": "low",
        "popularity": "high",
        "availability": "prefer_common",
        "longevity": "any",
        "sillage": "any",
        "budget_mode": "any",
    }


def test_explanation_uses_only_available_ranking_evidence() -> None:
    result = {
        "components": {
            "reminds_me_of": 0.76,
            "link_prediction": 0.84,
            "reference_content": 0.70,
            "season_fit": 0.81,
            "sweetness_fit": 0.91,
            "popularity": 0.95,
            "semantic_query": 0.88,
        },
        "community_evidence": {
            "cons": [
                {
                    "text": "May feel too generic",
                    "agreement": 0.72,
                    "up_votes": 18,
                    "down_votes": 7,
                }
            ]
        },
    }

    explanation = ExplanationGenerator().explain(
        _intent(), result, "Versace Pour Homme — Versace"
    )

    reason_types = {item["type"] for item in explanation["why_recommended"]}
    assert "community_similarity" in reason_types
    assert "ml_similarity" in reason_types
    assert explanation["evidence_level"] == "high"
    assert explanation["possible_drawbacks"][0]["votes"] == 25


def test_explanation_does_not_invent_missing_ml_or_community_signal() -> None:
    result = {
        "components": {"semantic_query": 0.67},
        "community_evidence": {"cons": []},
    }

    explanation = ExplanationGenerator().explain(_intent(), result)

    assert [item["type"] for item in explanation["why_recommended"]] == [
        "semantic_match"
    ]
    assert explanation["evidence_level"] == "limited"


def test_intent_summary_is_compact_and_localized() -> None:
    assert summarize_intent(_intent()) == [
        "мужской",
        "лета",
        "несладкий",
        "популярный",
    ]


def test_intent_summary_includes_budget_constraint() -> None:
    intent = _intent()
    intent["budget_mode"] = "affordable"
    assert summarize_intent(intent)[-1] == "недорогой"
