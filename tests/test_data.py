from pathlib import Path

import pandas as pd
import pytest

from scentmatch.data.audit import audit_profiles
from scentmatch.data.download import extract_required, sha256
from scentmatch.data.fragrantica import (
    extract_fragrance_id,
    normalize_identifier,
    split_values,
)
from scentmatch.data.select import proportional_quotas
from scentmatch.data.validate import require_columns


def test_extract_fragrance_id() -> None:
    url = "https://www.fragrantica.com/perfume/X/Y-74630.html"
    assert extract_fragrance_id(url) == "74630"


def test_extract_fragrance_id_rejects_invalid_value() -> None:
    assert extract_fragrance_id("not-a-fragrantica-url") is None


def test_normalize_identifier_ignores_case_and_punctuation() -> None:
    assert normalize_identifier("Eau-de Parfum") == "eaudeparfum"


def test_split_values_removes_unknown_and_empty_values() -> None:
    assert split_values("Rose, , unknown, Cedar, rose") == ["Rose", "Cedar"]


def test_proportional_quotas_sum_to_target() -> None:
    counts = pd.Series({"women | floral": 60, "men | woody": 30, "other": 10})
    quotas = proportional_quotas(counts, 37)
    assert int(quotas.sum()) == 37
    assert (quotas <= counts.loc[quotas.index]).all()


def test_profile_audit_accepts_complete_profile() -> None:
    profile = {
        "schema_version": "1.0",
        "perfume_id": "74630",
        "name": "Example",
        "brand": "Example Brand",
        "source_url": "https://example.test/74630",
        "gender": "unisex",
        "country": "France",
        "year": 2022,
        "rating": {"value": 4.1, "count": 500},
        "perfumers": [],
        "accords": ["woody"],
        "notes": {
            "top": ["Bergamot"],
            "middle": ["Rose"],
            "base": ["Cedar"],
        },
        "source_description": "A" * 80,
        "retrieval_text": "Example retrieval document",
    }
    report = audit_profiles([profile], expected_count=1)
    assert report["status"] == "passed"
    assert report["unique_perfume_ids"] == 1


def test_require_columns_reports_missing_columns() -> None:
    frame = pd.DataFrame({"name": ["Example"]})
    with pytest.raises(ValueError, match="brand"):
        require_columns(frame, {"name", "brand"}, "example.csv")


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_bytes(b"scentmatch")
    assert sha256(path) == (
        "5468f61db563da0084272578e22f6e31af2d84ac746ab703a7dd6060c98ed330"
    )


def test_extract_required_rejects_incomplete_archive(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fra_cleaned.csv", "url;Perfume\n")
    with pytest.raises(RuntimeError, match="fra_perfumes.csv"):
        extract_required(archive, tmp_path / "output")
