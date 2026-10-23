"""Download the selected Fragrantica dataset and record its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

DATASET_PAGE = (
    "https://www.kaggle.com/datasets/olgagmiufana1/"
    "fragrantica-com-fragrance-dataset"
)
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "olgagmiufana1/fragrantica-com-fragrance-dataset"
)
REQUIRED_MEMBERS = ("fra_cleaned.csv", "fra_perfumes.csv")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Download *url* atomically to *destination*."""
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ScentMatch/0.1 dataset downloader"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as target,
        ):
            shutil.copyfileobj(response, target)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_required(archive: Path, output_dir: Path) -> list[Path]:
    """Extract the two expected CSV files without accepting arbitrary paths."""
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = set(REQUIRED_MEMBERS) - names
        if missing:
            raise RuntimeError(f"Dataset archive is missing files: {sorted(missing)}")
        for member in REQUIRED_MEMBERS:
            destination = output_dir / member
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(destination)
    return extracted


def write_manifest(archive: Path, files: list[Path], output_dir: Path) -> Path:
    """Write local provenance and checksums for the downloaded version."""
    manifest = {
        "dataset": "Fragrantica.com Fragrance Dataset",
        "publisher": "Olga G (Miufana1)",
        "dataset_page": DATASET_PAGE,
        "download_url": DOWNLOAD_URL,
        "declared_license": "CC BY-NC-SA 4.0",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive": {
            "name": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "files": {
            file.name: {
                "size_bytes": file.stat().st_size,
                "sha256": sha256(file),
            }
            for file in files
        },
    }
    destination = output_dir / "dataset_manifest.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded raw files (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local download.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "fragrantica_dataset.zip"

    existing = [output_dir / name for name in REQUIRED_MEMBERS]
    if not args.force and archive.exists() and all(path.exists() for path in existing):
        print(f"Dataset already exists in {output_dir}")
        print("Use --force to download it again.")
        return

    print(f"Downloading dataset from {DATASET_PAGE}")
    download_file(DOWNLOAD_URL, archive)
    files = extract_required(archive, output_dir)
    manifest = write_manifest(archive, files, output_dir)
    print(f"Downloaded and extracted dataset to {output_dir}")
    print(f"Wrote provenance manifest to {manifest}")


if __name__ == "__main__":
    main()
