from scentmatch.evaluation.compare import compare_rankings


def test_compare_rankings_detects_improvement() -> None:
    queries = [
        {
            "query_id": "q001",
            "query_text": "example query",
            "split": "test",
            "category": "note",
        }
    ]
    baseline = {
        "q001": {
            "results": [
                {"perfume_id": "bad", "name": "Bad", "brand": "B", "silver_relevance": 0}
            ]
        }
    }
    finetuned = {
        "q001": {
            "results": [
                {"perfume_id": "good", "name": "Good", "brand": "G", "silver_relevance": 3}
            ]
        }
    }
    rows, report = compare_rankings(
        queries, baseline, finetuned, {"q001": {"good": 3}}
    )
    assert rows[0]["outcome"] == "improved"
    assert report["outcomes_by_ndcg_at_10"]["improved"] == 1
