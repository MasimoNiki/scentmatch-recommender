from scentmatch.evaluation.blind_v4 import _render_html, build_blind_pool


def _result(perfume_id: str, rank: int) -> dict:
    return {
        "perfume_id": perfume_id,
        "rank": rank,
        "name": f"Perfume {perfume_id}",
        "brand": "Brand",
        "accords": ["fresh"],
        "notes": {},
        "components": {"semantic_query": 0.9},
        "community_evidence": {},
    }


def test_blind_pool_deduplicates_and_hides_system_names() -> None:
    queries = [
        {
            "query_id": "q1",
            "query_text": "fresh summer scent",
            "category": "season",
            "split": "test",
        }
    ]
    rankings = {
        "semantic": [
            {"query_id": "q1", "results": [_result("p1", 1), _result("p2", 2)]}
        ],
        "community": [
            {"query_id": "q1", "results": [_result("p1", 1), _result("p3", 2)]}
        ],
    }

    public, secret = build_blind_pool(queries, rankings, per_system_depth=2)

    ids = {row["perfume_id"] for row in public["queries"][0]["candidates"]}
    assert ids == {"p1", "p2", "p3"}
    assert "systems" not in public["queries"][0]
    assert secret["queries"][0]["candidate_provenance"]["p1"] == {
        "community": 1,
        "semantic": 1,
    }


def test_html_is_standalone_and_exports_jsonl() -> None:
    public = {
        "schema_version": "4.0",
        "queries": [
            {
                "query_id": "q1",
                "query_text": "query",
                "category": "general",
                "candidates": [],
            }
        ],
    }

    rendered = _render_html(public)

    assert "<!doctype html>" in rendered
    assert "judgments_v4.jsonl" in rendered
    assert "localStorage" in rendered
    assert "join('\\n')+'\\n'" in rendered
