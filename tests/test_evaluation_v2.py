from scentmatch.evaluation.v2 import evaluate_intents, evaluate_references


def test_intent_evaluation_scores_only_declared_expected_fields() -> None:
    cases = [
        {
            "case_id": "intent-test",
            "query_text": "Мужской аромат на лето",
            "expected": {"gender": "men", "seasons": ["summer"]},
        }
    ]
    result = evaluate_intents(cases)
    assert result["case_accuracy"] == 1.0
    assert result["field_accuracy"] == 1.0


def test_reference_evaluation_measures_resolution_and_abstention() -> None:
    profiles = [
        {
            "perfume_id": "2318",
            "name": "versace-pour-homme",
            "brand": "versace",
        },
        {"perfume_id": "2", "name": "homme", "brand": "example"},
    ]
    cases = [
        {
            "case_id": "positive",
            "query_text": "Похожий на Vercahe Pour homme",
            "should_resolve": True,
            "expected_perfume_id": "2318",
        },
        {
            "case_id": "negative",
            "query_text": "Похожий на Homme",
            "should_resolve": False,
            "expected_perfume_id": None,
        },
    ]
    result = evaluate_references(cases, profiles)
    assert result["top1_accuracy"] == 1.0
    assert result["abstain_accuracy"] == 1.0
