from scentmatch.evaluation.silver import silver_relevance


def test_silver_relevance_honors_required_and_excluded_notes() -> None:
    query = {
        "constraints": {
            "gender": "any",
            "required_notes": ["bergamot"],
            "preferred_notes": ["musk"],
            "excluded_notes": ["vanilla"],
            "preferred_accords": ["fresh"],
            "excluded_accords": ["sweet"],
        }
    }
    relevant = {
        "gender": "unisex",
        "notes": {"top": ["bergamot"], "base": ["musk"]},
        "accords": ["fresh"],
    }
    excluded = {
        "gender": "unisex",
        "notes": {"top": ["bergamot"], "base": ["vanilla"]},
        "accords": ["fresh"],
    }
    assert silver_relevance(query, relevant) == 3
    assert silver_relevance(query, excluded) == 0
