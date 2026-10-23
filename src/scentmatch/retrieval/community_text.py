"""Build and query a compact index of vote-backed community statements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scentmatch.data.audit import read_jsonl

DEFAULT_COMMUNITY_PROFILES = Path("data/augmentation/community_profiles_v2.jsonl")
DEFAULT_ARTIFACT_DIR = Path("artifacts/community_text_v1")
DEFAULT_MODEL = Path("models/e5_finetuned")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            'ML dependencies are missing. Run: pip install -e ".[ml]"'
        ) from error
    return SentenceTransformer(model_name, device=device)


def _statement_rows(
    community_rows: list[dict[str, Any]],
    max_pros: int = 3,
    max_cons: int = 2,
) -> list[dict[str, Any]]:
    """Select reliable statements while retaining their user-vote provenance."""
    output: list[dict[str, Any]] = []
    for row in community_rows:
        summary = row.get("community_summary", {})
        for polarity, limit in (("pro", max_pros), ("con", max_cons)):
            values = summary.get(f"{polarity}s", [])
            selected = 0
            for statement in values:
                up_votes = max(0, int(statement.get("up_votes") or 0))
                down_votes = max(0, int(statement.get("down_votes") or 0))
                support = up_votes + down_votes
                wilson = float(statement.get("wilson_lower") or 0.0)
                text = str(statement.get("text") or "").strip()
                if not text or support < 5 or wilson < 0.60:
                    continue
                output.append(
                    {
                        "statement_id": (
                            f"{row['perfume_id']}:{polarity}:{selected + 1}"
                        ),
                        "perfume_id": str(row["perfume_id"]),
                        "polarity": polarity,
                        "text": text,
                        "up_votes": up_votes,
                        "down_votes": down_votes,
                        "agreement": round(up_votes / support, 6),
                        "wilson_lower": round(wilson, 6),
                    }
                )
                selected += 1
                if selected >= limit:
                    break
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_statement_index(args: argparse.Namespace) -> None:
    community_path = args.community_profiles.resolve()
    artifact_dir = args.artifact_dir.resolve()
    embeddings_path = artifact_dir / "statement_embeddings.npy"
    statements_path = artifact_dir / "statements.jsonl"
    metadata_path = artifact_dir / "metadata.json"
    if any(
        path.exists() for path in (embeddings_path, statements_path, metadata_path)
    ) and not args.force:
        raise FileExistsError(
            f"Artifacts already exist in {artifact_dir}. Use --force to rebuild."
        )

    community_rows = read_jsonl(community_path)
    statements = _statement_rows(
        community_rows,
        max_pros=args.max_pros,
        max_cons=args.max_cons,
    )
    model = _load_model(str(args.model), args.device)
    embeddings = model.encode(
        [f"passage: {row['text']}" for row in statements],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float16")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    _write_jsonl(statements_path, statements)
    metadata = {
        "schema_version": "1.0",
        "model": str(args.model),
        "statement_count": len(statements),
        "perfume_count": len({row["perfume_id"] for row in statements}),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "community_profiles_sha256": _sha256_file(community_path),
        "statements_sha256": _sha256_file(statements_path),
        "selection": {
            "minimum_votes": 5,
            "minimum_wilson_lower": 0.60,
            "max_pros_per_perfume": args.max_pros,
            "max_cons_per_perfume": args.max_cons,
        },
        "warning": (
            "Statements are source-provided AI summaries backed by user votes; "
            "they are evidence features, not ground truth."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


@dataclass(frozen=True)
class PerceptionMatch:
    score: float
    raw_similarity: float
    statement: dict[str, Any]


class CommunityTextIndex:
    """Statement-level cross-lingual retrieval grouped back to perfumes."""

    def __init__(
        self,
        embeddings: np.ndarray,
        statements: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        if embeddings.shape[0] != len(statements):
            raise ValueError("Statement embeddings and metadata are not aligned")
        self.embeddings = embeddings.astype("float32")
        self.statements = statements
        self.metadata = metadata

    @classmethod
    def load(cls, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> CommunityTextIndex:
        metadata = json.loads(
            (artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        statements_path = artifact_dir / "statements.jsonl"
        if metadata["statements_sha256"] != _sha256_file(statements_path):
            raise ValueError("Community statements changed after indexing")
        return cls(
            np.load(artifact_dir / "statement_embeddings.npy"),
            read_jsonl(statements_path),
            metadata,
        )

    def score(self, query_embedding: np.ndarray) -> dict[str, PerceptionMatch]:
        vector = np.asarray(query_embedding, dtype="float32").reshape(-1)
        similarities = self.embeddings @ vector
        best: dict[str, tuple[float, int]] = {}
        for index, (statement, similarity) in enumerate(
            zip(self.statements, similarities, strict=True)
        ):
            reliability = 1.0 - math.exp(
                -(statement["up_votes"] + statement["down_votes"]) / 30.0
            )
            evidence_score = float(similarity) * (
                0.70 + 0.20 * statement["wilson_lower"] + 0.10 * reliability
            )
            perfume_id = str(statement["perfume_id"])
            if perfume_id not in best or evidence_score > best[perfume_id][0]:
                best[perfume_id] = (evidence_score, index)

        ordered = sorted(best, key=lambda key: (best[key][0], key))
        denominator = max(1, len(ordered) - 1)
        percentiles = {
            perfume_id: rank / denominator
            for rank, perfume_id in enumerate(ordered)
        }
        return {
            perfume_id: PerceptionMatch(
                score=percentiles[perfume_id],
                raw_similarity=float(similarities[index]),
                statement=self.statements[index],
            )
            for perfume_id, (_, index) in best.items()
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--community-profiles", type=Path, default=DEFAULT_COMMUNITY_PROFILES
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pros", type=int, default=3)
    parser.add_argument("--max-cons", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    build_statement_index(build_parser().parse_args())


if __name__ == "__main__":
    main()
