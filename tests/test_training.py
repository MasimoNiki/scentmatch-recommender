from scentmatch.training.generate import _make_query
from scentmatch.training.mine import violation_reason


def test_generated_query_constraints_match_positive_profile() -> None:
    profile = {
        "perfume_id": "1",
        "accords": ["citrus", "fresh"],
        "notes": {
            "top": ["bergamot"],
            "middle": ["jasmine"],
            "base": ["musk"],
        },
    }
    generated = _make_query(profile, "negation", template_index=0)
    assert generated is not None
    text, constraints = generated
    assert "бергамот" in text
    assert constraints["required_notes"] == ["bergamot"]
    assert constraints["excluded_notes"]


def test_violation_reason_rejects_missing_required_note() -> None:
    query = {
        "constraints": {
            "required_notes": ["bergamot"],
            "excluded_notes": [],
            "preferred_accords": ["citrus"],
        }
    }
    profile = {
        "notes": {"top": ["lemon"], "middle": [], "base": []},
        "accords": ["citrus"],
    }
    assert violation_reason(query, profile) == "missing_required_note"
