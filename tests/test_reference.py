from scentmatch.retrieval.reference import (
    extract_reference_text,
    gender_compatible,
    has_budget_intent,
    resolve_reference,
    resolve_reference_detailed,
)

PROFILES = [
    {"perfume_id": "9099", "name": "bleu-de-chanel", "brand": "chanel"},
    {
        "perfume_id": "25967",
        "name": "bleu-de-chanel-eau-de-parfum",
        "brand": "chanel",
    },
    {"perfume_id": "31861", "name": "sauvage", "brand": "dior"},
    {
        "perfume_id": "2318",
        "name": "versace-pour-homme",
        "brand": "versace",
    },
    {"perfume_id": "999", "name": "homme", "brand": "david-beckham"},
]


def test_resolves_english_perfume_name_inside_russian_query() -> None:
    match = resolve_reference(
        "аромат похожий на Bleu de Chanel, но бюджетнее", PROFILES
    )
    assert match is not None
    assert match["perfume_id"] == "9099"


def test_longer_named_variant_wins() -> None:
    match = resolve_reference(
        "something like Bleu de Chanel Eau de Parfum but fresher", PROFILES
    )
    assert match is not None
    assert match["perfume_id"] == "25967"


def test_budget_intent_is_bilingual() -> None:
    assert has_budget_intent("похожий аромат, но бюджетнее")
    assert has_budget_intent("недорогой мужской аромат")
    assert has_budget_intent("similar but cheaper")
    assert has_budget_intent("a more affordable alternative")
    assert not has_budget_intent("fresh woody fragrance")


def test_generic_note_is_not_resolved_as_single_word_perfume() -> None:
    profiles = [
        {"perfume_id": "4", "name": "vanilla", "brand": "alyssa-ashley"}
    ]
    assert resolve_reference("fresh citrus fragrance with no vanilla", profiles) is None


def test_named_reference_preserves_gender_or_allows_unisex() -> None:
    reference = {"gender": "men"}
    assert gender_compatible(reference, {"gender": "men"})
    assert gender_compatible(reference, {"gender": "unisex"})
    assert not gender_compatible(reference, {"gender": "women"})


def test_extracts_only_reference_span() -> None:
    assert (
        extract_reference_text("Похожий на Bleu de Chanel, но более свежий")
        == "Bleu de Chanel"
    )


def test_typo_does_not_fall_back_to_generic_homme() -> None:
    result = resolve_reference_detailed(
        "Нужен популярный несладкий мужской парфюм "
        "похожий на Vercahe Pour homme",
        PROFILES,
    )
    assert result.accepted
    assert result.profile is not None
    assert result.profile["perfume_id"] == "2318"
    assert result.confidence >= 0.78


def test_ambiguous_generic_reference_abstains() -> None:
    result = resolve_reference_detailed("Ищу аромат похожий на Homme", PROFILES)
    assert not result.accepted
    assert result.profile is None
