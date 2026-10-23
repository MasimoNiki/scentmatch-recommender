from scentmatch.retrieval.intent import parse_query_intent

PROFILES = [
    {
        "perfume_id": "2318",
        "name": "versace-pour-homme",
        "brand": "versace",
    },
    {"perfume_id": "999", "name": "homme", "brand": "david-beckham"},
]


def test_parses_user_example_into_structured_intent() -> None:
    intent = parse_query_intent(
        "Нужен популярный несладкий мужской парфюм "
        "похожий на Vercahe Pour homme",
        PROFILES,
    )
    assert intent.gender == "men"
    assert intent.sweetness == "low"
    assert intent.popularity == "high"
    assert intent.availability == "prefer_common"
    assert intent.reference is not None
    assert intent.reference.accepted
    assert intent.reference.profile is not None
    assert intent.reference.profile["perfume_id"] == "2318"


def test_parses_season_occasion_and_explicit_exclusion() -> None:
    intent = parse_query_intent(
        "Ищу женский аромат на лето для офиса без ванили и табака"
    )
    assert intent.gender == "women"
    assert intent.seasons == ("summer",)
    assert intent.occasions == ("office",)
    assert intent.excluded_notes == ("tobacco", "vanilla")
    assert "explicit_exclusions" in intent.hard_constraints


def test_low_sweetness_wins_over_word_sweet_inside_negation() -> None:
    intent = parse_query_intent("A popular men's perfume that is not too sweet")
    assert intent.gender == "men"
    assert intent.sweetness == "low"
    assert intent.popularity == "high"


def test_parses_max_price_and_currency() -> None:
    intent = parse_query_intent("Мужской аромат до 5000 рублей")
    assert intent.budget_mode == "max_price"
    assert intent.max_price == 5000.0
    assert intent.currency == "RUB"


def test_parses_affordable_without_reference() -> None:
    intent = parse_query_intent(
        "Мужской недорогой аромат на осень с пачули и ветивером"
    )
    assert intent.budget_mode == "affordable"
    assert intent.reference is None
