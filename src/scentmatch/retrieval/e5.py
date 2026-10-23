"""Build, evaluate and query the multilingual-e5-small CPU baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scentmatch.data.audit import read_jsonl
from scentmatch.evaluation.metrics import evaluate_rankings
from scentmatch.retrieval.community_ranker import (
    DEFAULT_COMMUNITY_PROFILES,
    CommunityFeatureStore,
    percentile_features,
    query_dependent_weights,
    weighted_score,
)
from scentmatch.retrieval.constraints import (
    extract_exclusions,
    violates_exclusions,
)
from scentmatch.retrieval.intent import gender_matches_intent, parse_query_intent
from scentmatch.retrieval.reference import (
    display_name,
    gender_compatible,
    load_optional_index,
)
from scentmatch.training.link_prediction import CommunityLinkPredictor

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_PROFILES = Path("data/processed/perfume_profiles_5000.jsonl")
DEFAULT_QUERIES = Path("data/eval/queries_v1.jsonl")
DEFAULT_QRELS = Path("data/eval/silver_qrels_v1.jsonl")
DEFAULT_ARTIFACT_DIR = Path("artifacts/e5_small")
DEFAULT_SEARCH_ARTIFACT_DIR = Path("artifacts/e5_finetuned")
DEFAULT_COMMUNITY_SIGNALS = Path("data/augmentation/community_signals.jsonl")
DEFAULT_PRICE_SNAPSHOT = Path("data/augmentation/price_snapshot_2024.jsonl")
DEFAULT_REVIEW_ARTIFACT_DIR = Path("artifacts/e5_finetuned_reviews")
DEFAULT_LINK_MODEL = Path("artifacts/link_predictor_v1/model.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            'ML dependencies are missing. Run: pip install -e ".[ml]"'
        ) from error
    return SentenceTransformer(model_name, device=device)


def rank_by_similarity(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: list[str],
    top_k: int,
) -> tuple[list[list[str]], list[list[float]]]:
    """Rank normalized document vectors by cosine similarity."""
    if document_embeddings.shape[0] != len(document_ids):
        raise ValueError("Document IDs and embedding rows are not aligned")
    similarities = query_embeddings @ document_embeddings.T
    limit = min(top_k, len(document_ids))
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :limit]
    ranked_ids = [[document_ids[index] for index in row] for row in indices]
    ranked_scores = [
        [float(similarities[row_index, index]) for index in row]
        for row_index, row in enumerate(indices)
    ]
    return ranked_ids, ranked_scores


def build_embeddings(args: argparse.Namespace) -> None:
    profiles_path = args.profiles.resolve()
    artifact_dir = args.artifact_dir.resolve()
    embeddings_path = artifact_dir / "document_embeddings.npy"
    metadata_path = artifact_dir / "metadata.json"
    if (embeddings_path.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError(
            f"Artifacts already exist in {artifact_dir}. Use --force to rebuild."
        )

    profiles = read_jsonl(profiles_path)
    passages = [f"passage: {profile['retrieval_text']}" for profile in profiles]
    model = load_model(args.model, args.device)
    embeddings = model.encode(
        passages,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    metadata = {
        "model": args.model,
        "device_used": args.device,
        "document_count": len(profiles),
        "embedding_dimension": int(embeddings.shape[1]),
        "profiles_sha256": sha256_file(profiles_path),
        "perfume_ids": [str(profile["perfume_id"]) for profile in profiles],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("E5 document embeddings created.")
    print(f"Documents: {len(profiles)}")
    print(f"Dimension: {embeddings.shape[1]}")
    print(f"Embeddings: {embeddings_path}")


def _load_artifacts(
    artifact_dir: Path, profiles_path: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    embeddings = np.load(artifact_dir / "document_embeddings.npy")
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["profiles_sha256"] != sha256_file(profiles_path):
        raise ValueError("Profiles changed after embeddings were built. Rebuild them.")
    if embeddings.shape[0] != len(metadata["perfume_ids"]):
        raise ValueError("Embedding artifact and metadata are not aligned")
    return embeddings, metadata


def _load_review_artifacts(
    artifact_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    embeddings_path = artifact_dir / "review_embeddings.npy"
    metadata_path = artifact_dir / "metadata.json"
    if not embeddings_path.exists() or not metadata_path.exists():
        return None
    embeddings = np.load(embeddings_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(metadata["perfume_ids"]):
        raise ValueError("Review embedding artifact and metadata are not aligned")
    return embeddings, metadata


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    return {
        row["query_id"]: {key: int(value) for key, value in row["labels"].items()}
        for row in read_jsonl(path)
    }


def evaluate(args: argparse.Namespace) -> None:
    profiles_path = args.profiles.resolve()
    profiles = read_jsonl(profiles_path)
    queries = read_jsonl(args.queries.resolve())
    embeddings, metadata = _load_artifacts(
        args.artifact_dir.resolve(), profiles_path
    )
    model = load_model(metadata["model"], args.device)
    query_embeddings = model.encode(
        [f"query: {query['query_text']}" for query in queries],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    retrieval_depth = (
        min(len(profiles), args.ranking_depth + 250)
        if args.apply_exclusions
        else args.ranking_depth
    )
    ranked_ids, ranked_scores = rank_by_similarity(
        query_embeddings,
        embeddings,
        metadata["perfume_ids"],
        top_k=retrieval_depth,
    )
    if args.apply_exclusions:
        profiles_by_id = {
            str(profile["perfume_id"]): profile for profile in profiles
        }
        filtered_ids = []
        filtered_scores = []
        for query, ids, scores in zip(
            queries, ranked_ids, ranked_scores, strict=True
        ):
            exclusions = extract_exclusions(query["query_text"])
            kept = [
                (perfume_id, score)
                for perfume_id, score in zip(ids, scores, strict=True)
                if not violates_exclusions(profiles_by_id[perfume_id], exclusions)
            ][: args.ranking_depth]
            filtered_ids.append([perfume_id for perfume_id, _ in kept])
            filtered_scores.append([score for _, score in kept])
        ranked_ids, ranked_scores = filtered_ids, filtered_scores
    rankings = {
        query["query_id"]: ids for query, ids in zip(queries, ranked_ids, strict=True)
    }
    qrels = _load_qrels(args.qrels.resolve())
    metrics: dict[str, Any] = {"all": evaluate_rankings(rankings, qrels)}
    for split in ("dev", "test"):
        split_ids = {query["query_id"] for query in queries if query["split"] == split}
        metrics[split] = evaluate_rankings(
            {key: value for key, value in rankings.items() if key in split_ids},
            {key: value for key, value in qrels.items() if key in split_ids},
        )

    profiles_by_id = {str(profile["perfume_id"]): profile for profile in profiles}
    rankings_path = args.rankings.resolve()
    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    with rankings_path.open("w", encoding="utf-8", newline="\n") as stream:
        for query, ids, scores in zip(
            queries, ranked_ids, ranked_scores, strict=True
        ):
            results = []
            for perfume_id, score in zip(ids, scores, strict=True):
                profile = profiles_by_id[perfume_id]
                results.append(
                    {
                        "perfume_id": perfume_id,
                        "score": round(score, 6),
                        "name": profile["name"],
                        "brand": profile["brand"],
                        "silver_relevance": qrels[query["query_id"]].get(
                            perfume_id, 0
                        ),
                    }
                )
            stream.write(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        "query_text": query["query_text"],
                        "split": query["split"],
                        "results": results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "model": metadata["model"],
        "device": args.device,
        "documents": len(profiles),
        "queries": len(queries),
        "qrels": "silver / automated_constraints_v1",
        "explicit_exclusion_filter": args.apply_exclusions,
        "metrics": metrics,
        "warning": "Metrics are proxy scores until human judgments exist.",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_path = args.review_csv.resolve()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "query_id",
            "query_text",
            "rank",
            "perfume_id",
            "name",
            "brand",
            "score",
            "silver_relevance",
            "accords",
            "top_notes",
            "middle_notes",
            "base_notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for query, ids, scores in zip(
            queries, ranked_ids, ranked_scores, strict=True
        ):
            for rank, (perfume_id, score) in enumerate(
                zip(ids[:5], scores[:5], strict=True), start=1
            ):
                profile = profiles_by_id[perfume_id]
                writer.writerow(
                    {
                        "query_id": query["query_id"],
                        "query_text": query["query_text"],
                        "rank": rank,
                        "perfume_id": perfume_id,
                        "name": profile["name"],
                        "brand": profile["brand"],
                        "score": round(score, 6),
                        "silver_relevance": qrels[query["query_id"]].get(
                            perfume_id, 0
                        ),
                        "accords": ", ".join(profile["accords"]),
                        "top_notes": ", ".join(profile["notes"]["top"]),
                        "middle_notes": ", ".join(profile["notes"]["middle"]),
                        "base_notes": ", ".join(profile["notes"]["base"]),
                    }
                )
    print("E5 baseline evaluation completed.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Rankings: {rankings_path}")
    print(f"Report: {report_path}")
    print(f"Review CSV: {review_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Create document embeddings.")
    build.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    build.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--device", default="cpu")
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument("--force", action="store_true")
    build.set_defaults(handler=build_embeddings)

    evaluation = subparsers.add_parser("evaluate", help="Evaluate query retrieval.")
    evaluation.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    evaluation.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    evaluation.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    evaluation.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.add_argument("--ranking-depth", type=int, default=100)
    evaluation.add_argument("--apply-exclusions", action="store_true")
    evaluation.add_argument(
        "--rankings", type=Path, default=Path("artifacts/e5_small/rankings.jsonl")
    )
    evaluation.add_argument(
        "--report", type=Path, default=Path("data/interim/e5_baseline_report.json")
    )
    evaluation.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/interim/e5_results_review.csv"),
    )
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


def _structured_fit(profile: dict[str, Any], preferences: set[str]) -> float | None:
    if not preferences:
        return None
    notes = {
        str(note).casefold()
        for values in profile["notes"].values()
        for note in values
    }
    accords = {str(value).casefold() for value in profile["accords"]}
    return len(preferences & (notes | accords)) / len(preferences)


def _diversify(
    candidates: list[dict[str, Any]],
    embeddings: np.ndarray,
    id_to_index: dict[str, int],
    top_k: int,
    has_reference: bool,
) -> list[dict[str, Any]]:
    """MMR-style selection plus a two-per-brand cap."""
    pool = candidates[: min(250, len(candidates))]
    selected: list[dict[str, Any]] = []
    brand_counts: dict[str, int] = {}
    penalty_weight = 0.02 if has_reference else 0.04
    while pool and len(selected) < top_k:
        best_index = 0
        best_adjusted = float("-inf")
        best_penalty = 0.0
        for index, candidate in enumerate(pool):
            brand = str(candidate["profile"]["brand"]).casefold()
            if brand_counts.get(brand, 0) >= 2:
                continue
            redundancy = 0.0
            if selected:
                vector = embeddings[id_to_index[candidate["perfume_id"]]]
                redundancy = max(
                    float(vector @ embeddings[id_to_index[item["perfume_id"]]])
                    for item in selected
                )
            penalty = penalty_weight * max(0.0, (redundancy - 0.75) / 0.25)
            adjusted = candidate["score"] - penalty
            if adjusted > best_adjusted:
                best_index = index
                best_adjusted = adjusted
                best_penalty = penalty
        chosen = pool.pop(best_index)
        chosen["diversity_adjustment"] = -best_penalty
        chosen["final_score"] = best_adjusted
        selected.append(chosen)
        brand = str(chosen["profile"]["brand"]).casefold()
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    return selected


def search_main() -> None:
    parser = argparse.ArgumentParser(description="Search perfumes with hybrid ranking.")
    parser.add_argument("query")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument(
        "--artifact-dir", type=Path, default=DEFAULT_SEARCH_ARTIFACT_DIR
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--community-signals", type=Path, default=DEFAULT_COMMUNITY_SIGNALS
    )
    parser.add_argument(
        "--community-profiles", type=Path, default=DEFAULT_COMMUNITY_PROFILES
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_SNAPSHOT)
    parser.add_argument(
        "--review-artifact-dir", type=Path, default=DEFAULT_REVIEW_ARTIFACT_DIR
    )
    parser.add_argument("--link-model", type=Path, default=DEFAULT_LINK_MODEL)
    parser.add_argument(
        "--no-link-model",
        action="store_true",
        help="Disable the optional community link-prediction feature.",
    )
    parser.add_argument(
        "--review-rerank",
        action="store_true",
        help="Enable the legacy experimental review similarity adjustment.",
    )
    parser.add_argument(
        "--no-review-rerank",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    profiles_path = args.profiles.resolve()
    profiles = read_jsonl(profiles_path)
    embeddings, metadata = _load_artifacts(args.artifact_dir.resolve(), profiles_path)
    model = load_model(metadata["model"], args.device)
    query_embedding = model.encode(
        [f"query: {args.query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    profiles_by_id = {str(profile["perfume_id"]): profile for profile in profiles}
    intent = parse_query_intent(args.query, profiles)
    reference_resolution = intent.reference
    reference = (
        reference_resolution.profile
        if reference_resolution is not None and reference_resolution.accepted
        else None
    )
    reference_id = str(reference["perfume_id"]) if reference is not None else None
    budget_intent = intent.budget_mode != "any"
    legacy_community = load_optional_index(args.community_signals.resolve())
    prices = load_optional_index(args.prices.resolve())
    community_path = args.community_profiles.resolve()
    community_store = (
        CommunityFeatureStore.load(community_path)
        if community_path.exists()
        else None
    )

    perfume_ids = [str(value) for value in metadata["perfume_ids"]]
    id_to_index = {perfume_id: index for index, perfume_id in enumerate(perfume_ids)}
    link_predictor = None
    link_model_path = args.link_model.resolve()
    if (
        not args.no_link_model
        and reference_id is not None
        and community_store is not None
        and link_model_path.exists()
    ):
        link_predictor = CommunityLinkPredictor.load(
            link_model_path,
            profiles,
            list(community_store.rows.values()),
            embeddings,
            perfume_ids,
        )
    semantic_raw_values = {
        perfume_id: float(score)
        for perfume_id, score in zip(
            perfume_ids, (query_embedding @ embeddings.T)[0], strict=True
        )
    }
    semantic_percentiles = percentile_features(semantic_raw_values)
    reference_raw_values: dict[str, float] = {}
    reference_percentiles: dict[str, float] = {}
    if reference_id is not None:
        reference_vector = embeddings[
            id_to_index[reference_id] : id_to_index[reference_id] + 1
        ]
        reference_raw_values = {
            perfume_id: float(score)
            for perfume_id, score in zip(
                perfume_ids, (reference_vector @ embeddings.T)[0], strict=True
            )
        }
        reference_percentiles = percentile_features(reference_raw_values)

    review_scores: dict[str, float] = {}
    review_adjustments: dict[str, float] = {}
    review_counts: dict[str, int] = {}
    review_artifacts = None
    if args.review_rerank and not args.no_review_rerank:
        review_artifacts = _load_review_artifacts(args.review_artifact_dir.resolve())
    if review_artifacts is not None:
        review_embeddings, review_metadata = review_artifacts
        chunk_scores = (query_embedding @ review_embeddings.T)[0]
        for perfume_id, score in zip(
            review_metadata["perfume_ids"], chunk_scores, strict=True
        ):
            review_scores[perfume_id] = max(
                review_scores.get(perfume_id, -1.0), float(score)
            )
        review_counts = {
            str(key): int(value)
            for key, value in review_metadata.get("review_counts", {}).items()
        }
        values = np.asarray(list(review_scores.values()), dtype="float32")
        if len(values) and float(values.std()) > 0:
            mean_score = float(values.mean())
            standard_deviation = float(values.std())
            review_adjustments = {
                perfume_id: 0.01
                * float(np.clip((score - mean_score) / standard_deviation, -2.0, 2.0))
                for perfume_id, score in review_scores.items()
            }

    exclusions = {
        "notes": set(intent.excluded_notes),
        "accords": set(intent.excluded_accords),
    }
    preferences = set(intent.preferred_notes) | set(intent.preferred_accords)
    weights = query_dependent_weights(intent, reference is not None)
    reference_price = prices.get(reference_id or "", {}).get("median_price")
    candidates = []
    gate_counts: dict[str, int] = {}
    for perfume_id in perfume_ids:
        if perfume_id == reference_id:
            continue
        candidate = profiles_by_id[perfume_id]
        if reference is not None and not gender_compatible(reference, candidate):
            continue
        if not gender_matches_intent(intent, candidate):
            continue
        if violates_exclusions(candidate, exclusions):
            continue
        if community_store is not None:
            gate = community_store.violates_community_gate(perfume_id, intent)
            if gate is not None:
                gate_counts[gate] = gate_counts.get(gate, 0) + 1
                continue
            community_features = community_store.features(
                perfume_id, intent, reference_id
            )
            components = community_features.to_dict()
            if link_predictor is not None and reference_id is not None:
                components["link_prediction"] = link_predictor.score(
                    reference_id, perfume_id
                )
        else:
            components = {}
            if intent.sweetness == "low" and "sweet" in {
                str(value).casefold() for value in candidate.get("accords", [])
            }:
                continue
        components.update(
            {
                "semantic_query": semantic_percentiles[perfume_id],
                "reference_content": reference_percentiles.get(perfume_id),
                "structured_fit": _structured_fit(candidate, preferences),
            }
        )
        score, used_weights = weighted_score(components, weights)
        score += review_adjustments.get(perfume_id, 0.0)
        if budget_intent and reference_price is not None:
            candidate_price = prices.get(perfume_id, {}).get("median_price")
            if candidate_price is None or candidate_price >= reference_price:
                continue
            budget_fit = 1.0 - candidate_price / reference_price
            components["budget_fit"] = budget_fit
            score += 0.02 * budget_fit
        elif budget_intent:
            if (
                legacy_community.get(perfume_id, {}).get("segment", "").casefold()
                != "mass-market"
            ):
                continue
            components["budget_fit"] = 1.0
        candidates.append(
            {
                "perfume_id": perfume_id,
                "profile": candidate,
                "score": score,
                "semantic_raw": semantic_raw_values[perfume_id],
                "reference_raw": reference_raw_values.get(perfume_id),
                "components": components,
                "weights": used_weights,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    kept = _diversify(
        candidates, embeddings, id_to_index, args.top_k, reference is not None
    )

    print(f"Query: {args.query}")
    if reference is not None:
        print(
            "Reference perfume: "
            f"{display_name(reference['name'])} — {display_name(reference['brand'])} "
            f"(confidence={reference_resolution.confidence:.3f}; "
            f"margin={reference_resolution.margin:.3f})"
        )
    elif reference_resolution is not None and reference_resolution.raw_text:
        print(
            "Reference perfume: unresolved "
            f"(text={reference_resolution.raw_text!r}; "
            f"confidence={reference_resolution.confidence:.3f}; "
            f"margin={reference_resolution.margin:.3f})"
        )
    if community_store is not None:
        print(
            "Hybrid ranking: E5 + community votes + community similarity graph "
            f"({len(community_store.rows)} profiles)"
        )
        print(
            "Active weights: "
            + ", ".join(f"{name}={value:.2f}" for name, value in weights.items())
        )
    else:
        print("Community v2 data not found; using a reduced embedding-only fallback.")
    if review_artifacts is not None:
        print(
            "Legacy review adjustment: explicitly enabled, experimental "
            f"({len(review_counts)} perfumes)"
        )
    if link_predictor is not None:
        print(
            "Community link predictor: enabled "
            "(portable logistic model; held-out edge evaluation v1)"
        )
    if gate_counts:
        print(
            "Community gates removed: "
            + ", ".join(f"{key}={value}" for key, value in sorted(gate_counts.items()))
        )
    if budget_intent:
        if reference_price is not None:
            print(
                f"Budget filter: below ${reference_price:.2f} "
                "(historical eBay snapshot, May 2024; not a live price)"
            )
        else:
            print("Budget filter: mass-market segment proxy; exact price unavailable")
    if exclusions["notes"] or exclusions["accords"]:
        print(
            "Applied exclusions: "
            + ", ".join(sorted(exclusions["notes"] | exclusions["accords"]))
        )
    if preferences:
        print("Structured preferences: " + ", ".join(sorted(preferences)))

    output_results = []
    for position, item in enumerate(kept, start=1):
        perfume_id = item["perfume_id"]
        profile = item["profile"]
        accords = ", ".join(profile["accords"][:5])
        components = item["components"]
        details = [
            f"score={item['final_score']:.4f}",
            f"semantic={item['semantic_raw']:.4f}",
            f"accords={accords}",
        ]
        for key, label in (
            ("season_fit", "season"),
            ("longevity_fit", "longevity"),
            ("sillage_fit", "sillage"),
            ("popularity", "popularity"),
            ("reminds_me_of", "smells-like"),
            ("also_liked", "also-liked"),
            ("link_prediction", "ml-similarity"),
        ):
            value = components.get(key)
            if value is not None and (
                key not in {"reminds_me_of", "also_liked"} or value > 0
            ):
                details.append(f"{label}={value:.3f}")
        if item["diversity_adjustment"]:
            details.append(f"diversity={item['diversity_adjustment']:+.3f}")
        consensus = community_store.consensus(perfume_id) if community_store else {}
        pros = consensus.get("pros", [])
        cons = consensus.get("cons", [])
        if pros:
            details.append(
                f"community+={pros[0]['text']} ({pros[0]['agreement']:.0%})"
            )
        if cons:
            details.append(
                f"community-={cons[0]['text']} ({cons[0]['agreement']:.0%})"
            )
        if perfume_id in review_scores:
            details.append(
                f"legacy-review={review_scores[perfume_id]:.4f}; "
                f"adjustment={review_adjustments.get(perfume_id, 0.0):+.4f}"
            )
        print(
            f"{position}. {display_name(profile['name'])} — "
            f"{display_name(profile['brand'])} ({'; '.join(details)})"
        )
        output_results.append(
            {
                "rank": position,
                "perfume_id": perfume_id,
                "name": display_name(profile["name"]),
                "brand": display_name(profile["brand"]),
                "score": round(float(item["final_score"]), 6),
                "base_score": round(float(item["score"]), 6),
                "components": {
                    key: round(float(value), 6) if value is not None else None
                    for key, value in components.items()
                },
                "weights": {
                    key: round(float(value), 6)
                    for key, value in item["weights"].items()
                },
                "community_evidence": {"pros": pros[:2], "cons": cons[:2]},
            }
        )
    if args.output_json is not None:
        output = {
            "schema_version": "3.0",
            "query": args.query,
            "intent": intent.to_dict(),
            "reference_id": reference_id,
            "ranking_mode": (
                "hybrid_community_v2" if community_store else "embedding_fallback"
            ),
            "active_weights": weights,
            "gate_counts": gate_counts,
            "results": output_results,
        }
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
