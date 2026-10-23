"""Build compact E5 embeddings for matched perfume-review chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scentmatch.data.audit import read_jsonl
from scentmatch.retrieval.e5 import load_model, sha256_file

DEFAULT_REVIEWS = Path("data/reviews/atrafshan_review_chunks.jsonl")
DEFAULT_ARTIFACT_DIR = Path("artifacts/e5_finetuned_reviews")
DEFAULT_MODEL = "models/e5_finetuned"


def build(args: argparse.Namespace) -> None:
    reviews_path = args.reviews.resolve()
    artifact_dir = args.artifact_dir.resolve()
    embeddings_path = artifact_dir / "review_embeddings.npy"
    metadata_path = artifact_dir / "metadata.json"
    if (embeddings_path.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError(
            f"Artifacts already exist in {artifact_dir}. Use --force to rebuild."
        )

    chunks = read_jsonl(reviews_path)
    model = load_model(str(args.model), args.device)
    embeddings = model.encode(
        [f"passage: {chunk['text']}" for chunk in chunks],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    review_counts: dict[str, int] = {}
    for chunk in chunks:
        perfume_id = str(chunk["perfume_id"])
        review_counts[perfume_id] = max(
            review_counts.get(perfume_id, 0),
            int(chunk["eligible_reviews_for_perfume"]),
        )
    metadata = {
        "model": str(args.model),
        "device_used": args.device,
        "chunk_count": len(chunks),
        "perfume_count": len(review_counts),
        "embedding_dimension": int(embeddings.shape[1]),
        "reviews_sha256": sha256_file(reviews_path),
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "perfume_ids": [str(chunk["perfume_id"]) for chunk in chunks],
        "review_counts": review_counts,
        "language": "fa",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Review embeddings created.")
    print(f"Chunks: {len(chunks)}")
    print(f"Perfumes: {len(review_counts)}")
    print(f"Embeddings: {embeddings_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    build_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    build_parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    build_parser.add_argument("--device", default="cpu")
    build_parser.add_argument("--batch-size", type=int, default=16)
    build_parser.add_argument("--force", action="store_true")
    build_parser.set_defaults(handler=build)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
