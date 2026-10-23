"""User-facing ScentMatch commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scentmatch.retrieval.community_ranker import DEFAULT_COMMUNITY_PROFILES
from scentmatch.retrieval.e5 import (
    DEFAULT_COMMUNITY_SIGNALS,
    DEFAULT_LINK_MODEL,
    DEFAULT_PRICE_SNAPSHOT,
    DEFAULT_PROFILES,
    DEFAULT_SEARCH_ARTIFACT_DIR,
)
from scentmatch.service import RecommendationService, ServiceSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend perfumes with hybrid ML/community ranking."
    )
    parser.add_argument("query")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument(
        "--artifact-dir", type=Path, default=DEFAULT_SEARCH_ARTIFACT_DIR
    )
    parser.add_argument(
        "--community-profiles", type=Path, default=DEFAULT_COMMUNITY_PROFILES
    )
    parser.add_argument(
        "--community-signals", type=Path, default=DEFAULT_COMMUNITY_SIGNALS
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_SNAPSHOT)
    parser.add_argument("--link-model", type=Path, default=DEFAULT_LINK_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-link-model", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def search_main() -> None:
    args = _parser().parse_args()
    service = RecommendationService(
        ServiceSettings(
            profiles_path=args.profiles,
            artifact_dir=args.artifact_dir,
            community_profiles_path=args.community_profiles,
            community_signals_path=args.community_signals,
            prices_path=args.prices,
            link_model_path=args.link_model,
            device=args.device,
            enable_link_model=not args.no_link_model,
        )
    )
    response = service.recommend(args.query, top_k=args.top_k)
    print(f"Query: {response['query']}")
    reference = response.get("reference")
    if reference and reference.get("display_name"):
        print(f"Reference perfume: {reference['display_name']}")
    elif reference and reference.get("raw_text"):
        print(f"Reference perfume: unresolved ({reference['raw_text']})")
    print(f"Link predictor: {response['link_predictor_enabled']}")
    for warning in response["warnings"]:
        print(f"Warning: {warning}")
    for result in response["results"]:
        explanation = result["explanation"]
        print(
            f"{result['rank']}. {result['name']} — {result['brand']} "
            f"(score={result['score']:.4f}; "
            f"evidence={explanation['evidence_level']})"
        )
        for reason in explanation["why_recommended"]:
            print(f"   + {reason['text']} [{reason['strength']}]")
        for drawback in explanation["possible_drawbacks"]:
            print(
                f"   - {drawback['text']} "
                f"(agreement={drawback['agreement']:.0%})"
            )

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    search_main()
