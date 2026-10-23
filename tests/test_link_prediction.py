import numpy as np

from scentmatch.training.link_prediction import (
    FEATURE_NAMES,
    PortableLogisticModel,
    aggregate_edges,
    canonical_pair,
    select_negative_edges,
    select_positive_edges,
    split_pair,
)


def test_reverse_edges_share_one_pair_and_one_split() -> None:
    rows = [
        {
            "perfume_id": "2",
            "similarity": {
                "reminds_me_of": [
                    {"perfume_id": "1", "up_votes": 8, "down_votes": 2}
                ]
            },
        },
        {
            "perfume_id": "1",
            "similarity": {
                "reminds_me_of": [
                    {"perfume_id": "2", "up_votes": 7, "down_votes": 3}
                ]
            },
        },
    ]

    edges = aggregate_edges(rows)

    assert edges == {("1", "2"): {"up_votes": 8, "down_votes": 3}}
    assert canonical_pair("2", "1") == ("1", "2")
    assert split_pair(("1", "2"), 42) == split_pair(("1", "2"), 42)


def test_only_explicit_vote_backed_labels_are_selected() -> None:
    edges = {
        ("1", "2"): {"up_votes": 9, "down_votes": 1},
        ("1", "3"): {"up_votes": 4, "down_votes": 6},
        ("1", "4"): {"up_votes": 0, "down_votes": 0},
    }

    assert select_positive_edges(edges) == {("1", "2")}
    assert select_negative_edges(edges) == {("1", "3")}


def test_portable_model_matches_logistic_formula() -> None:
    model = PortableLogisticModel(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.ones(len(FEATURE_NAMES)),
        intercept=0.0,
    )
    features = np.zeros((2, len(FEATURE_NAMES)), dtype="float32")
    features[1, 0] = 1.0

    probabilities = model.predict_proba(features)

    assert probabilities[0] == 0.5
    assert np.isclose(probabilities[1], 1.0 / (1.0 + np.exp(-1.0)))
