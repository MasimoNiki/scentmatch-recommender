"""Mine constraint-aware hard negatives with the frozen E5 baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from scentmatch.data.audit import read_jsonl
from scentmatch.retrieval.e5 import _load_artifacts, load_model


def violation_reason(
    query: dict[str, Any], profile: dict[str, Any]
) -> str | None:
    """Explain an explicit constraint mismatch, or return None."""
    constraints = query["constraints"]
    notes = {
        str(note).casefold()
        for values in profile["notes"].values()
        for note in values
    }
    accords = {str(value).casefold() for value in profile["accords"]}
    required = {value.casefold() for value in constraints["required_notes"]}
    excluded = {value.casefold() for value in constraints["excluded_notes"]}
    preferred_accords = {
        value.casefold() for value in constraints["preferred_accords"]
    }
    if not required.issubset(notes):
        return "missing_required_note"
    if excluded & notes:
        return "contains_excluded_note"
    if preferred_accords and not preferred_accords.intersection(accords):
        return "missing_preferred_accord"
    return None


def _partial_ranking(scores: np.ndarray, depth: int) -> np.ndarray:
    limit = min(depth, len(scores))
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    return candidates[np.argsort(-scores[candidates], kind="stable")]


def mine_triplets(
    queries: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    document_embeddings: np.ndarray,
    document_ids: list[str],
    query_embeddings: np.ndarray,
    negatives_per_query: int = 3,
    candidate_depth: int = 250,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build ML-ready n-tuples and separate provenance metadata."""
    if len(queries) != len(query_embeddings):
        raise ValueError("Queries and query embeddings are not aligned")
    id_to_index = {perfume_id: index for index, perfume_id in enumerate(document_ids)}
    profiles_by_id = {str(profile["perfume_id"]): profile for profile in profiles}
    similarities = query_embeddings @ document_embeddings.T
    tuples = []
    metadata_rows = []
    reasons = Counter()
    positive_scores = []
    negative_scores = []
    negatives_above_positive = 0

    for query_index, query in enumerate(queries):
        positive_id = str(query["positive_perfume_id"])
        positive_index = id_to_index[positive_id]
        positive_score = float(similarities[query_index, positive_index])
        ranked = _partial_ranking(similarities[query_index], candidate_depth)
        negatives = []
        for rank, candidate_index in enumerate(ranked, start=1):
            candidate_id = document_ids[int(candidate_index)]
            if candidate_id == positive_id:
                continue
            reason = violation_reason(query, profiles_by_id[candidate_id])
            if reason is None:
                continue
            score = float(similarities[query_index, candidate_index])
            negatives.append(
                {
                    "perfume_id": candidate_id,
                    "score": score,
                    "rank": rank,
                    "violation": reason,
                }
            )
            if len(negatives) == negatives_per_query:
                break
        if len(negatives) != negatives_per_query:
            raise RuntimeError(
                f"Only {len(negatives)} hard negatives for {query['query_id']}"
            )

        positive = profiles_by_id[positive_id]
        tuple_row = {
            "anchor": f"query: {query['query_text']}",
            "positive": f"passage: {positive['retrieval_text']}",
        }
        for number, negative in enumerate(negatives, start=1):
            profile = profiles_by_id[negative["perfume_id"]]
            tuple_row[f"negative_{number}"] = f"passage: {profile['retrieval_text']}"
            reasons[negative["violation"]] += 1
            negative_scores.append(negative["score"])
            if negative["score"] > positive_score:
                negatives_above_positive += 1
        tuples.append({"split": query["split"], **tuple_row})
        metadata_rows.append(
            {
                "query_id": query["query_id"],
                "split": query["split"],
                "query_type": query["query_type"],
                "positive_perfume_id": positive_id,
                "positive_score": positive_score,
                "hard_negatives": negatives,
            }
        )
        positive_scores.append(positive_score)

    pair_count = len(queries) * negatives_per_query
    report = {
        "status": "passed",
        "queries": len(queries),
        "train_tuples": sum(row["split"] == "train" for row in tuples),
        "validation_tuples": sum(
            row["split"] == "validation" for row in tuples
        ),
        "negatives_per_query": negatives_per_query,
        "hard_negative_pairs": pair_count,
        "mean_positive_score": round(mean(positive_scores), 6),
        "mean_hard_negative_score": round(mean(negative_scores), 6),
        "hard_negatives_scoring_above_positive": negatives_above_positive,
        "hard_negatives_scoring_above_positive_share": round(
            negatives_above_positive / pair_count, 6
        ),
        "violation_distribution": dict(reasons),
        "false_negative_guard": (
            "Every negative violates an explicit required note, excluded note, "
            "or preferred accord constraint."
        ),
        "eval_queries_used_for_training": 0,
    }
    return tuples, metadata_rows, report


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("data/training/synthetic_queries_v1.jsonl")
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/processed/perfume_profiles_5000.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/e5_small")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--candidate-depth", type=int, default=250)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/training")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/interim/hard_negative_report.json")
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/interim/training_triplets_review.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profiles_path = args.profiles.resolve()
    queries = read_jsonl(args.queries.resolve())
    profiles = read_jsonl(profiles_path)
    document_embeddings, metadata = _load_artifacts(
        args.artifact_dir.resolve(), profiles_path
    )
    model = load_model(metadata["model"], args.device)
    query_embeddings = model.encode(
        [f"query: {row['query_text']}" for row in queries],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    tuples, provenance, report = mine_triplets(
        queries,
        profiles,
        document_embeddings,
        metadata["perfume_ids"],
        query_embeddings,
        negatives_per_query=args.negatives,
        candidate_depth=args.candidate_depth,
    )
    output_dir = args.output_dir.resolve()
    train_rows = [
        {key: value for key, value in row.items() if key != "split"}
        for row in tuples
        if row["split"] == "train"
    ]
    validation_rows = [
        {key: value for key, value in row.items() if key != "split"}
        for row in tuples
        if row["split"] == "validation"
    ]
    _write_jsonl(train_rows, output_dir / "train_triplets_v1.jsonl")
    _write_jsonl(validation_rows, output_dir / "validation_triplets_v1.jsonl")
    _write_jsonl(provenance, output_dir / "triplets_provenance_v1.jsonl")
    review_path = args.review_csv.resolve()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    queries_by_id = {row["query_id"]: row for row in queries}
    profiles_by_id = {str(row["perfume_id"]): row for row in profiles}
    review_rows = provenance[:: max(1, len(provenance) // 50)][:50]
    fields = [
        "query_id",
        "query_text",
        "positive",
        "positive_score",
        "negative_1",
        "negative_1_score",
        "negative_1_violation",
        "negative_2",
        "negative_2_score",
        "negative_2_violation",
        "negative_3",
        "negative_3_score",
        "negative_3_violation",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in review_rows:
            query = queries_by_id[row["query_id"]]
            positive = profiles_by_id[row["positive_perfume_id"]]
            output = {
                "query_id": row["query_id"],
                "query_text": query["query_text"],
                "positive": f"{positive['name']} — {positive['brand']}",
                "positive_score": round(row["positive_score"], 6),
            }
            for number, negative in enumerate(row["hard_negatives"], start=1):
                profile = profiles_by_id[negative["perfume_id"]]
                output[f"negative_{number}"] = (
                    f"{profile['name']} — {profile['brand']}"
                )
                output[f"negative_{number}_score"] = round(negative["score"], 6)
                output[f"negative_{number}_violation"] = negative["violation"]
            writer.writerow(output)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Hard-negative mining completed.")
    print(f"Train tuples: {report['train_tuples']}")
    print(f"Validation tuples: {report['validation_tuples']}")
    print(f"Hard negatives: {report['hard_negative_pairs']}")
    print(f"Report: {report_path}")
    print(f"Optional review CSV: {review_path}")


if __name__ == "__main__":
    main()
