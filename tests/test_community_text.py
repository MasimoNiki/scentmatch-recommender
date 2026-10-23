import numpy as np

from scentmatch.retrieval.community_text import CommunityTextIndex, _statement_rows


def test_statement_selection_requires_vote_backed_confidence() -> None:
    rows = [
        {
            "perfume_id": "p1",
            "community_summary": {
                "pros": [
                    {
                        "text": "Fresh and discreet office scent",
                        "up_votes": 20,
                        "down_votes": 1,
                        "wilson_lower": 0.8,
                    },
                    {
                        "text": "Unsupported",
                        "up_votes": 2,
                        "down_votes": 0,
                        "wilson_lower": 0.2,
                    },
                ],
                "cons": [],
            },
        }
    ]

    statements = _statement_rows(rows)

    assert [row["text"] for row in statements] == [
        "Fresh and discreet office scent"
    ]
    assert statements[0]["agreement"] == 0.952381


def test_statement_index_returns_best_evidence_per_perfume() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype="float32"
    )
    statements = [
        {
            "perfume_id": "p1",
            "text": "fresh",
            "up_votes": 20,
            "down_votes": 0,
            "wilson_lower": 0.8,
        },
        {
            "perfume_id": "p1",
            "text": "office",
            "up_votes": 10,
            "down_votes": 1,
            "wilson_lower": 0.7,
        },
        {
            "perfume_id": "p2",
            "text": "heavy",
            "up_votes": 20,
            "down_votes": 0,
            "wilson_lower": 0.8,
        },
    ]
    index = CommunityTextIndex(embeddings, statements, {})

    matches = index.score(np.asarray([1.0, 0.0], dtype="float32"))

    assert matches["p1"].statement["text"] == "fresh"
    assert matches["p1"].score == 1.0
    assert matches["p2"].score == 0.0
