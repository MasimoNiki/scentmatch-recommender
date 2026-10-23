from scentmatch.data.reviews import prepare_review_chunks


def _source_row(text: str, *, reply: str = "NO", vote: str = "8") -> dict:
    return {
        "perfume_id": "source-1",
        "perfume_name": "Bleu de Chanel",
        "perfume_brand": "Chanel",
        "comment_text": text,
        "user_id": "private-user",
        "user_name": "private-name",
        "user_vote_on_perfume": {
            "votes": {"رایحه": vote},
            "is_comment_answer": reply,
        },
    }


def test_prepare_review_chunks_matches_and_removes_identity() -> None:
    profiles = [
        {
            "perfume_id": "9099",
            "name": "bleu-de-chanel",
            "brand": "chanel",
        }
    ]
    rows = [
        _source_row(
            "A sufficiently detailed review describing citrus, woods and incense."
        )
    ]
    chunks, report = prepare_review_chunks(rows, profiles)
    assert report["matched_catalog_perfumes"] == 1
    assert chunks[0]["perfume_id"] == "9099"
    assert "user_id" not in chunks[0]
    assert "user_name" not in chunks[0]


def test_prepare_review_chunks_filters_replies_and_low_votes() -> None:
    profiles = [
        {
            "perfume_id": "9099",
            "name": "bleu-de-chanel",
            "brand": "chanel",
        }
    ]
    text = "A sufficiently detailed perfume review that should be excluded."
    rows = [_source_row(text, reply="YES"), _source_row(text, vote="3")]
    chunks, report = prepare_review_chunks(rows, profiles)
    assert chunks == []
    assert report["eligible_standalone_review_rows"] == 0
