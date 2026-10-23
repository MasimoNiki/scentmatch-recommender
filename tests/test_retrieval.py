import numpy as np

from scentmatch.retrieval.constraints import (
    extract_exclusions,
    extract_preferences,
    preference_bonus,
    violates_exclusions,
)
from scentmatch.retrieval.e5 import rank_by_similarity


def test_rank_by_similarity_returns_best_dot_product() -> None:
    queries = np.array([[1.0, 0.0]], dtype="float32")
    documents = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
    ids, scores = rank_by_similarity(queries, documents, ["a", "b"], top_k=2)
    assert ids == [["b", "a"]]
    assert scores[0][0] == 1.0


def test_explicit_russian_exclusions_filter_profile() -> None:
    exclusions = extract_exclusions("Свежий аромат без ванили и табака")
    assert exclusions["notes"] == {"vanilla", "tobacco"}
    profile = {
        "notes": {"top": [], "middle": [], "base": ["vanilla"]},
        "accords": ["sweet"],
    }
    assert violates_exclusions(profile, exclusions)


def test_explicit_english_exclusions_filter_profile() -> None:
    exclusions = extract_exclusions("Fresh woody fragrance with no vanilla")
    assert exclusions["notes"] == {"vanilla"}
    profile = {
        "notes": {"top": [], "middle": [], "base": []},
        "accords": ["vanilla", "woody"],
    }
    assert violates_exclusions(profile, exclusions)


def test_bilingual_positive_preferences_ignore_negated_note() -> None:
    preferences = extract_preferences(
        "A dark smoky woody fragrance with incense, without vanilla"
    )
    assert preferences == {"smoky", "woody", "incense"}
    profile = {
        "notes": {"top": [], "middle": ["incense"], "base": []},
        "accords": ["smoky", "woody"],
    }
    assert preference_bonus(profile, preferences) == 0.12
