from pathlib import Path

from scentmatch.runtime import REQUIRED_FILES, missing_runtime_assets


def test_runtime_check_reports_missing_files(tmp_path: Path) -> None:
    missing = missing_runtime_assets(tmp_path)

    assert set(REQUIRED_FILES).issubset(missing)
    assert "models/e5_finetuned/*.safetensors" in missing


def test_runtime_check_accepts_complete_layout(tmp_path: Path) -> None:
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "models/e5_finetuned/model.safetensors").touch()

    assert missing_runtime_assets(tmp_path) == []
