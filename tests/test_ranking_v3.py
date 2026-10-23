from scentmatch.evaluation.ranking_v3 import evaluate_ranking_rows, exposure_gini


def test_human_metrics_and_quality_gates() -> None:
    queries = [{"query_id": "q1", "category": "season"}]
    rankings = [
        {
            "query_id": "q1",
            "results": [
                {"perfume_id": "1", "brand": "A"},
                {"perfume_id": "2", "brand": "B"},
            ],
        }
    ]
    judgments = [
        {
            "query_id": "q1",
            "perfume_id": "1",
            "relevance": 3,
            "constraint_violation": False,
            "evidence_quality": 2,
        },
        {
            "query_id": "q1",
            "perfume_id": "2",
            "relevance": 2,
            "constraint_violation": False,
            "evidence_quality": 2,
        },
    ]
    report = evaluate_ranking_rows(queries, rankings, judgments, k=2)
    assert report["complete"]
    assert report["overall"]["ndcg_at_k"] == 1.0
    assert report["overall"]["constraint_violation_rate"] == 0.0


def test_missing_judgment_makes_report_incomplete() -> None:
    queries = [{"query_id": "q1", "category": "season"}]
    rankings = [
        {"query_id": "q1", "results": [{"perfume_id": "1", "brand": "A"}]}
    ]
    report = evaluate_ranking_rows(queries, rankings, [], k=1)
    assert not report["complete"]
    assert report["missing_judgments"]


def test_exposure_gini_is_zero_for_uniform_counts() -> None:
    assert exposure_gini([2, 2, 2]) == 0.0
