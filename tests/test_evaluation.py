from scentmatch.evaluation.validate import validate_queries


def test_validate_queries_accepts_known_terms() -> None:
    profiles = [
        {
            "notes": {"top": ["bergamot"], "middle": ["rose"], "base": ["musk"]},
            "accords": ["citrus", "fresh"],
        }
    ]
    queries = [
        {
            "schema_version": "1.0",
            "query_id": "q001",
            "query_text": "Ищу свежий аромат с заметным бергамотом.",
            "language": "ru",
            "split": "dev",
            "category": "note",
            "difficulty": "easy",
            "constraints": {
                "gender": "any",
                "required_notes": ["bergamot"],
                "preferred_notes": [],
                "excluded_notes": [],
                "preferred_accords": ["citrus"],
                "excluded_accords": [],
                "style_tags": ["свежий"],
            },
            "annotation_status": "unreviewed",
        }
    ]
    report = validate_queries(queries, profiles, expected_count=1)
    assert report["status"] == "passed"
    assert report["unique_query_ids"] == 1


def test_validate_queries_rejects_unknown_note() -> None:
    profiles = [{"notes": {"top": ["rose"]}, "accords": ["floral"]}]
    queries = [
        {
            "schema_version": "1.0",
            "query_id": "q001",
            "query_text": "Ищу аромат с несуществующей тестовой нотой.",
            "language": "ru",
            "split": "test",
            "category": "note",
            "difficulty": "easy",
            "constraints": {
                "gender": "any",
                "required_notes": ["missing-note"],
                "preferred_notes": [],
                "excluded_notes": [],
                "preferred_accords": [],
                "excluded_accords": [],
                "style_tags": [],
            },
            "annotation_status": "unreviewed",
        }
    ]
    report = validate_queries(queries, profiles, expected_count=1)
    assert report["status"] == "failed"
    assert report["errors"]["unknown_note_terms"] == ["q001: missing-note"]
