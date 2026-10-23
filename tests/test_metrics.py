import pytest

from scentmatch.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ranked_metrics() -> None:
    ranking = ["a", "b", "c"]
    labels = {"a": 0, "b": 3, "c": 2, "d": 2}
    assert recall_at_k(ranking, labels, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranking, labels, 3) == pytest.approx(0.5)
    assert precision_at_k(ranking, labels, 3) == pytest.approx(2 / 3)
    assert hit_rate_at_k(ranking, labels, 1) == 0
    assert hit_rate_at_k(ranking, labels, 2) == 1
    assert 0 < ndcg_at_k(ranking, labels, 3) < 1
