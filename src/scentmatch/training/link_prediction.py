"""Train a leakage-safe community link predictor for reference recommendations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scentmatch.data.audit import read_jsonl

DEFAULT_PROFILES = Path("data/processed/perfume_profiles_5000.jsonl")
DEFAULT_COMMUNITY = Path("data/augmentation/community_profiles_v2.jsonl")
DEFAULT_EMBEDDINGS = Path("artifacts/e5_finetuned/document_embeddings.npy")
DEFAULT_METADATA = Path("artifacts/e5_finetuned/metadata.json")
DEFAULT_ARTIFACT = Path("artifacts/link_predictor_v1/model.json")
FEATURE_NAMES = (
    "content_cosine",
    "accord_jaccard",
    "note_jaccard",
    "accord_strength_cosine",
    "season_cosine",
    "daypart_cosine",
    "longevity_similarity",
    "sillage_similarity",
    "rating_similarity",
    "popularity_min",
    "popularity_max",
    "also_liked",
    "common_neighbor_jaccard",
    "adamic_adar",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))  # type: ignore[return-value]


def _pair_fraction(pair: tuple[str, str], seed: int) -> float:
    key = f"{seed}:{pair[0]}:{pair[1]}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return value / (2**64 - 1)


def split_pair(pair: tuple[str, str], seed: int) -> str:
    value = _pair_fraction(pair, seed)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "test"


def aggregate_edges(
    community_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, int]]:
    """Collapse directions so reverse duplicates can never cross a split."""
    output: dict[tuple[str, str], dict[str, int]] = {}
    for row in community_rows:
        source_id = str(row["perfume_id"])
        for edge in row.get("similarity", {}).get("reminds_me_of", []):
            target_id = str(edge["perfume_id"])
            if source_id == target_id:
                continue
            pair = canonical_pair(source_id, target_id)
            current = output.setdefault(pair, {"up_votes": 0, "down_votes": 0})
            current["up_votes"] = max(
                current["up_votes"], max(0, int(edge.get("up_votes") or 0))
            )
            current["down_votes"] = max(
                current["down_votes"], max(0, int(edge.get("down_votes") or 0))
            )
    return output


def select_positive_edges(
    edges: dict[tuple[str, str], dict[str, int]],
    minimum_votes: int = 10,
    minimum_agreement: float = 0.70,
) -> set[tuple[str, str]]:
    return {
        pair
        for pair, votes in edges.items()
        if votes["up_votes"] + votes["down_votes"] >= minimum_votes
        and votes["up_votes"]
        / max(1, votes["up_votes"] + votes["down_votes"])
        >= minimum_agreement
    }


def select_negative_edges(
    edges: dict[tuple[str, str], dict[str, int]],
    minimum_votes: int = 10,
    maximum_agreement: float = 0.40,
) -> set[tuple[str, str]]:
    """Use only pairs explicitly rejected by the community, not missing edges."""
    return {
        pair
        for pair, votes in edges.items()
        if votes["up_votes"] + votes["down_votes"] >= minimum_votes
        and votes["up_votes"]
        / max(1, votes["up_votes"] + votes["down_votes"])
        <= maximum_agreement
    }


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _cosine(first: Iterable[float], second: Iterable[float]) -> float:
    left = np.asarray(list(first), dtype="float32")
    right = np.asarray(list(second), dtype="float32")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right) / denominator if denominator else 0.0


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    denominator = max(1, len(ordered) - 1)
    return {key: rank / denominator for rank, key in enumerate(ordered)}


class PairFeatureStore:
    """Compute content, community and train-graph features without target edges."""

    def __init__(
        self,
        profiles: list[dict[str, Any]],
        community_rows: list[dict[str, Any]],
        embeddings: np.ndarray,
        perfume_ids: list[str],
        train_edges: set[tuple[str, str]],
    ) -> None:
        self.profiles = {str(row["perfume_id"]): row for row in profiles}
        self.community = {
            str(row["perfume_id"]): row for row in community_rows
        }
        self.embeddings = embeddings.astype("float32")
        self.id_to_index = {
            str(perfume_id): index for index, perfume_id in enumerate(perfume_ids)
        }
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        for first, second in train_edges:
            self.adjacency[first].add(second)
            self.adjacency[second].add(first)
        self.also_liked = {
            perfume_id: {
                str(value)
                for value in row.get("similarity", {}).get("also_liked", [])
            }
            for perfume_id, row in self.community.items()
        }
        self.note_sets = {
            perfume_id: {
                str(note).casefold()
                for values in profile.get("notes", {}).values()
                for note in values
            }
            for perfume_id, profile in self.profiles.items()
        }
        self.accord_sets = {
            perfume_id: {
                str(value).casefold() for value in profile.get("accords", [])
            }
            for perfume_id, profile in self.profiles.items()
        }
        popularity_raw = {}
        for perfume_id, row in self.community.items():
            popularity = row.get("popularity", {})
            popularity_raw[perfume_id] = (
                0.50 * math.log1p(max(0, int(popularity.get("magnitude") or 0)))
                + 0.30
                * math.log1p(
                    max(0, int(popularity.get("compound_magnitude") or 0))
                )
                + 0.20
                * math.log1p(
                    max(0, int(popularity.get("recent_magnitude") or 0))
                )
            )
        self.popularity = _percentiles(popularity_raw)

    def _notes(self, perfume_id: str) -> set[str]:
        return self.note_sets[perfume_id]

    def _accords(self, perfume_id: str) -> set[str]:
        return self.accord_sets[perfume_id]

    def _vote_vector(
        self, perfume_id: str, field: str, keys: tuple[str, ...]
    ) -> list[float]:
        values = self.community[perfume_id].get(field, {})
        return [max(0.0, float(values.get(key) or 0.0)) for key in keys]

    def _accord_strength_vector(
        self, first: str, second: str
    ) -> tuple[list[float], list[float]]:
        first_values = self.community[first].get("accord_strengths", {})
        second_values = self.community[second].get("accord_strengths", {})
        keys = sorted(set(first_values) | set(second_values))
        return (
            [float(first_values.get(key) or 0.0) for key in keys],
            [float(second_values.get(key) or 0.0) for key in keys],
        )

    def features(self, first: str, second: str) -> np.ndarray:
        first = str(first)
        second = str(second)
        left_index = self.id_to_index[first]
        right_index = self.id_to_index[second]
        common = self.adjacency[first] & self.adjacency[second]
        union = self.adjacency[first] | self.adjacency[second]
        adamic = sum(
            1.0 / max(math.log1p(len(self.adjacency[node])), 1.0)
            for node in common
        )
        accord_left, accord_right = self._accord_strength_vector(first, second)
        first_community = self.community[first]
        second_community = self.community[second]
        longevity_difference = abs(
            float(first_community.get("longevity", {}).get("average") or 3.0)
            - float(second_community.get("longevity", {}).get("average") or 3.0)
        )
        sillage_difference = abs(
            float(first_community.get("sillage", {}).get("average") or 2.5)
            - float(second_community.get("sillage", {}).get("average") or 2.5)
        )
        rating_difference = abs(
            float(first_community.get("rating", {}).get("average") or 0.0)
            - float(second_community.get("rating", {}).get("average") or 0.0)
        )
        popularity = sorted(
            (self.popularity.get(first, 0.0), self.popularity.get(second, 0.0))
        )
        values = [
            float(self.embeddings[left_index] @ self.embeddings[right_index]),
            _jaccard(self._accords(first), self._accords(second)),
            _jaccard(self._notes(first), self._notes(second)),
            _cosine(accord_left, accord_right),
            _cosine(
                self._vote_vector(
                    first, "seasons", ("winter", "spring", "summer", "autumn")
                ),
                self._vote_vector(
                    second, "seasons", ("winter", "spring", "summer", "autumn")
                ),
            ),
            _cosine(
                self._vote_vector(first, "daypart", ("day", "night")),
                self._vote_vector(second, "daypart", ("day", "night")),
            ),
            1.0 - min(1.0, longevity_difference / 4.0),
            1.0 - min(1.0, sillage_difference / 3.0),
            1.0 - min(1.0, rating_difference / 5.0),
            popularity[0],
            popularity[1],
            float(
                second in self.also_liked.get(first, set())
                or first in self.also_liked.get(second, set())
            ),
            len(common) / len(union) if union else 0.0,
            min(1.0, adamic / 5.0),
        ]
        return np.asarray(values, dtype="float32")


@dataclass(frozen=True)
class PortableLogisticModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(features).astype("float64")
        standardized = (matrix - self.mean) / self.scale
        logits = standardized @ self.coefficients + self.intercept
        logits = np.clip(logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> PortableLogisticModel:
        model = artifact["model"]
        return cls(
            feature_names=tuple(model["feature_names"]),
            mean=np.asarray(model["mean"], dtype="float64"),
            scale=np.asarray(model["scale"], dtype="float64"),
            coefficients=np.asarray(model["coefficients"], dtype="float64"),
            intercept=float(model["intercept"]),
        )


class CommunityLinkPredictor:
    """Inference-only wrapper; it does not import scikit-learn."""

    def __init__(
        self,
        artifact: dict[str, Any],
        profiles: list[dict[str, Any]],
        community_rows: list[dict[str, Any]],
        embeddings: np.ndarray,
        perfume_ids: list[str],
    ) -> None:
        self.model = PortableLogisticModel.from_artifact(artifact)
        if self.model.feature_names != FEATURE_NAMES:
            raise ValueError("Link predictor feature schema is incompatible")
        train_edges = {
            canonical_pair(*pair) for pair in artifact["graph"]["train_edges"]
        }
        self.store = PairFeatureStore(
            profiles,
            community_rows,
            embeddings,
            perfume_ids,
            train_edges,
        )

    @classmethod
    def load(
        cls,
        artifact_path: Path,
        profiles: list[dict[str, Any]],
        community_rows: list[dict[str, Any]],
        embeddings: np.ndarray,
        perfume_ids: list[str],
    ) -> CommunityLinkPredictor:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        return cls(
            artifact,
            profiles,
            community_rows,
            embeddings,
            perfume_ids,
        )

    def score(self, reference_id: str, candidate_id: str) -> float:
        features = self.store.features(reference_id, candidate_id)
        return float(self.model.predict_proba(features)[0])


def _dataset(
    positives: set[tuple[str, str]],
    negatives: set[tuple[str, str]],
    split: str,
    store: PairFeatureStore,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    pairs = sorted(
        pair
        for pair in positives | negatives
        if split_pair(pair, seed) == split
    )
    rows = [store.features(*pair) for pair in pairs]
    labels = [int(pair in positives) for pair in pairs]
    return np.vstack(rows), np.asarray(labels, dtype="int8"), pairs


def _pairwise_ranking_metrics(
    positives: set[tuple[str, str]],
    negatives: set[tuple[str, str]],
    split: str,
    store: PairFeatureStore,
    model: PortableLogisticModel,
    seed: int,
) -> dict[str, dict[str, float]]:
    positive_by_source: dict[str, set[str]] = defaultdict(set)
    negative_by_source: dict[str, set[str]] = defaultdict(set)
    for pair in positives:
        if split_pair(pair, seed) == split:
            positive_by_source[pair[0]].add(pair[1])
            positive_by_source[pair[1]].add(pair[0])
    for pair in negatives:
        if split_pair(pair, seed) == split:
            negative_by_source[pair[0]].add(pair[1])
            negative_by_source[pair[1]].add(pair[0])

    shared_sources = sorted(set(positive_by_source) & set(negative_by_source))
    results: dict[str, list[float]] = {
        "content_cosine": [],
        "ml": [],
    }
    query_results: dict[str, list[float]] = {
        "content_cosine": [],
        "ml": [],
    }
    top1_results: dict[str, list[float]] = {
        "content_cosine": [],
        "ml": [],
    }
    for source in shared_sources:
        positives_for_source = sorted(positive_by_source[source])
        negatives_for_source = sorted(negative_by_source[source])
        candidates = positives_for_source + negatives_for_source
        matrix = np.vstack(
            [store.features(source, candidate) for candidate in candidates]
        )
        scores = {
            "content_cosine": matrix[:, FEATURE_NAMES.index("content_cosine")],
            "ml": model.predict_proba(matrix),
        }
        positive_count = len(positives_for_source)
        for name, values in scores.items():
            comparisons = []
            for positive_score in values[:positive_count]:
                for negative_score in values[positive_count:]:
                    comparisons.append(
                        float(positive_score > negative_score)
                        + 0.5 * float(positive_score == negative_score)
                    )
            results[name].extend(comparisons)
            query_results[name].append(sum(comparisons) / len(comparisons))
            top1_results[name].append(
                float(int(np.argmax(values)) < positive_count)
            )

    def summarize(name: str) -> dict[str, float]:
        return {
            "queries": len(shared_sources),
            "pair_comparisons": len(results[name]),
            "pairwise_accuracy": round(float(np.mean(results[name])), 6),
            "macro_pairwise_accuracy": round(
                float(np.mean(query_results[name])), 6
            ),
            "top1_positive_rate": round(float(np.mean(top1_results[name])), 6),
        }

    return {name: summarize(name) for name in results}


def train(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    profiles_path = args.profiles.resolve()
    community_path = args.community.resolve()
    embeddings_path = args.embeddings.resolve()
    metadata_path = args.metadata.resolve()
    profiles = read_jsonl(profiles_path)
    community_rows = read_jsonl(community_path)
    embeddings = np.load(embeddings_path).astype("float32")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    perfume_ids = [str(value) for value in metadata["perfume_ids"]]
    if embeddings.shape[0] != len(perfume_ids):
        raise ValueError("Embedding rows and perfume IDs are not aligned")

    aggregated = aggregate_edges(community_rows)
    positives = select_positive_edges(
        aggregated, args.minimum_votes, args.minimum_agreement
    )
    negatives = select_negative_edges(
        aggregated, args.minimum_votes, args.maximum_negative_agreement
    )
    if positives & negatives:
        raise ValueError("Positive and negative edge sets overlap")
    train_edges = {pair for pair in positives if split_pair(pair, args.seed) == "train"}
    store = PairFeatureStore(
        profiles, community_rows, embeddings, perfume_ids, train_edges
    )
    datasets = {
        split: _dataset(
            positives,
            negatives,
            split,
            store,
            args.seed,
        )
        for split in ("train", "dev", "test")
    }
    train_x, train_y, _ = datasets["train"]
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=args.regularization,
        random_state=args.seed,
    ).fit(scaler.transform(train_x), train_y)
    portable = PortableLogisticModel(
        feature_names=FEATURE_NAMES,
        mean=scaler.mean_,
        scale=np.where(scaler.scale_ == 0, 1.0, scaler.scale_),
        coefficients=classifier.coef_[0],
        intercept=float(classifier.intercept_[0]),
    )
    classification = {}
    for split in ("dev", "test"):
        features, labels, _ = datasets[split]
        probabilities = portable.predict_proba(features)
        classification[split] = {
            "pairs": len(labels),
            "positive_pairs": int(labels.sum()),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
            "average_precision": round(
                float(average_precision_score(labels, probabilities)), 6
            ),
        }
    ranking = {
        split: _pairwise_ranking_metrics(
            positives,
            negatives,
            split,
            store,
            portable,
            args.seed,
        )
        for split in ("dev", "test")
    }
    coefficients = sorted(
        zip(FEATURE_NAMES, classifier.coef_[0], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    artifact = {
        "schema_version": "1.0",
        "task": "community_reminds_me_of_link_prediction",
        "model": {
            "type": "standardized_logistic_regression",
            "feature_names": list(FEATURE_NAMES),
            "mean": [round(float(value), 10) for value in scaler.mean_],
            "scale": [round(float(value), 10) for value in portable.scale],
            "coefficients": [
                round(float(value), 10) for value in classifier.coef_[0]
            ],
            "intercept": round(float(classifier.intercept_[0]), 10),
        },
        "graph": {
            "split_seed": args.seed,
            "minimum_votes": args.minimum_votes,
            "minimum_agreement": args.minimum_agreement,
            "maximum_negative_agreement": args.maximum_negative_agreement,
            "observed_pairs": len(aggregated),
            "positive_pairs": len(positives),
            "negative_pairs": len(negatives),
            "train_edges": [list(pair) for pair in sorted(train_edges)],
        },
        "datasets": {
            split: {
                "rows": len(values[1]),
                "positives": int(values[1].sum()),
                "negatives": int((values[1] == 0).sum()),
            }
            for split, values in datasets.items()
        },
        "classification": classification,
        "ranking": ranking,
        "largest_standardized_coefficients": [
            {"feature": name, "coefficient": round(float(value), 6)}
            for name, value in coefficients[:10]
        ],
        "data": {
            "profiles_sha256": _sha256(profiles_path),
            "community_sha256": _sha256(community_path),
            "embeddings_sha256": _sha256(embeddings_path),
        },
        "leakage_control": (
            "Undirected pairs are split deterministically; dev/test reminds-me-of "
            "edges are absent from graph features. Negatives are only observed "
            "pairs explicitly rejected by community votes; missing edges are not "
            "treated as negative labels."
        ),
        "ablation_decision": (
            "Node degree features were excluded because dev ablation showed hub "
            "bias. Common-neighbor features use train-positive edges only."
        ),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in ("datasets", "classification", "ranking")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--community", type=Path, default=DEFAULT_COMMUNITY)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--minimum-votes", type=int, default=10)
    parser.add_argument("--minimum-agreement", type=float, default=0.70)
    parser.add_argument("--maximum-negative-agreement", type=float, default=0.40)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
