from scentmatch.retrieval.community_ranker import (
    CommunityFeatureStore,
    percentile_features,
    query_dependent_weights,
)
from scentmatch.retrieval.intent import parse_query_intent


def _row(
    perfume_id: str,
    *,
    summer: int,
    winter: int,
    sweet: float,
    popularity: int,
) -> dict:
    return {
        "perfume_id": perfume_id,
        "seasons": {
            "winter": winter,
            "spring": 10,
            "summer": summer,
            "autumn": 10,
        },
        "daypart": {"day": 40, "night": 20},
        "accord_strengths": {"sweet": sweet},
        "rating": {"average": 4.0, "votes": 100},
        "longevity": {"average": 3.5, "votes": 50},
        "sillage": {"average": 2.5, "votes": 50},
        "community_gender": {"male": 80, "unisex": 20},
        "popularity": {
            "magnitude": popularity,
            "compound_magnitude": popularity,
            "recent_magnitude": popularity,
        },
        "similarity": {"reminds_me_of": [], "also_liked": []},
        "community_summary": {"pros": [], "cons": []},
    }


def test_summer_votes_beat_a_winter_profile() -> None:
    store = CommunityFeatureStore(
        [
            _row("1", summer=80, winter=10, sweet=0.0, popularity=100),
            _row("2", summer=5, winter=90, sweet=0.0, popularity=100),
        ]
    )
    assert store.season_fit("1", ("summer",)) > store.season_fit(
        "2", ("summer",)
    )


def test_explicit_summer_gate_rejects_supported_mismatch() -> None:
    store = CommunityFeatureStore(
        [
            _row("1", summer=2, winter=98, sweet=0.0, popularity=100),
            _row("2", summer=80, winter=10, sweet=0.0, popularity=200),
        ]
    )
    intent = parse_query_intent("мужской аромат на лето")
    assert store.violates_community_gate("1", intent) == "season_mismatch"


def test_low_sweetness_uses_accord_strength_not_presence_only() -> None:
    store = CommunityFeatureStore(
        [
            _row("1", summer=50, winter=10, sweet=0.8, popularity=100),
            _row("2", summer=50, winter=10, sweet=0.2, popularity=200),
        ]
    )
    intent = parse_query_intent("несладкий аромат")
    assert store.violates_community_gate("1", intent) == "too_sweet"
    assert store.violates_community_gate("2", intent) is None


def test_reference_graph_is_vote_aware_and_bidirectional() -> None:
    first = _row("1", summer=50, winter=10, sweet=0.0, popularity=100)
    second = _row("2", summer=50, winter=10, sweet=0.0, popularity=200)
    second["similarity"]["reminds_me_of"] = [
        {"perfume_id": "1", "up_votes": 90, "down_votes": 10}
    ]
    store = CommunityFeatureStore([first, second])
    forward, _ = store.reference_graph_scores("1", "2")
    assert forward > 0.5


def test_reference_and_season_queries_activate_different_weights() -> None:
    season = parse_query_intent("мужской аромат на лето")
    reference = parse_query_intent("мужской аромат похожий на Bleu de Chanel")
    season_weights = query_dependent_weights(season, has_reference=False)
    reference_weights = query_dependent_weights(reference, has_reference=True)
    assert "season_fit" in season_weights
    assert "reminds_me_of" in reference_weights
    assert "link_prediction" in reference_weights
    assert "reminds_me_of" not in season_weights


def test_percentiles_put_highest_value_at_one() -> None:
    assert percentile_features({"a": 0.1, "b": 0.5, "c": 0.3}) == {
        "a": 0.0,
        "c": 0.5,
        "b": 1.0,
    }
