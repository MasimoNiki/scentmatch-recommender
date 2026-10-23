"""Validate local runtime assets required by the recommendation service."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_FILES = (
    "data/processed/perfume_profiles_5000.jsonl",
    "data/augmentation/community_profiles_v2.jsonl",
    "data/augmentation/community_signals.jsonl",
    "data/augmentation/price_snapshot_2024.jsonl",
    "artifacts/e5_finetuned/document_embeddings.npy",
    "artifacts/e5_finetuned/metadata.json",
    "artifacts/link_predictor_v1/model.json",
    "models/e5_finetuned/config.json",
    "models/e5_finetuned/modules.json",
)


def missing_runtime_assets(root: Path) -> list[str]:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    model_dir = root / "models/e5_finetuned"
    if not any(model_dir.glob("*.safetensors")):
        missing.append("models/e5_finetuned/*.safetensors")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    missing = missing_runtime_assets(args.root.resolve())
    if missing:
        print("ScentMatch runtime assets are incomplete:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)
    print("ScentMatch runtime assets: OK")


if __name__ == "__main__":
    main()
