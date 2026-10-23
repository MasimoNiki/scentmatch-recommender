import pytest

from scentmatch.service import RecommendationService


class FakeEngine:
    def __init__(self) -> None:
        self.profiles_by_id = {
            "1": {
                "perfume_id": "1",
                "name": "reference-one",
                "brand": "brand-a",
                "accords": ["fresh"],
            },
            "2": {
                "perfume_id": "2",
                "name": "candidate-two",
                "brand": "brand-b",
                "accords": ["citrus"],
            },
        }
        self.metadata = {
            "model": "models/e5_finetuned",
            "embedding_dimension": 384,
        }
        self.link_predictor = object()
        self.community_store = None

    def rank(self, query: str, mode: str, top_k: int) -> dict:
        return {
            "intent": {
                "query_text": query,
                "language": "ru",
                "gender": "any",
                "seasons": [],
                "occasions": [],
                "preferred_notes": [],
                "excluded_notes": [],
                "preferred_accords": [],
                "excluded_accords": [],
                "sweetness": "any",
                "popularity": "any",
                "availability": "any",
                "longevity": "any",
                "sillage": "any",
                "budget_mode": "any",
                "max_price": None,
                "currency": None,
                "reference": {
                    "raw_text": "Reference One",
                    "accepted": True,
                    "confidence": 0.95,
                },
            },
            "reference_id": "1",
            "link_predictor_enabled": True,
            "gate_counts": {},
            "results": [
                {
                    "rank": 1,
                    "perfume_id": "2",
                    "name": "Candidate Two",
                    "brand": "Brand B",
                    "score": 0.82,
                    "components": {
                        "semantic_query": 0.8,
                        "reference_content": 0.7,
                        "link_prediction": 0.81,
                    },
                    "weights": {"semantic_query": 0.2},
                    "accords": ["citrus"],
                    "notes": {},
                    "gender": "unisex",
                    "year": 2020,
                    "source_url": "https://example.test/perfume/2",
                    "community_evidence": {"pros": [], "cons": []},
                }
            ][:top_k],
        }


def test_service_enriches_ranker_results_without_second_model_call() -> None:
    service = RecommendationService(engine=FakeEngine())

    response = service.recommend("Похожий на Reference One", top_k=1)

    assert response["reference"]["display_name"] == "Reference One — Brand A"
    assert response["results"][0]["explanation"]["why_recommended"]
    assert response["link_predictor_enabled"]


def test_service_validates_public_input_contract() -> None:
    service = RecommendationService(engine=FakeEngine())

    with pytest.raises(ValueError, match="must not be empty"):
        service.recommend("   ")
    with pytest.raises(ValueError, match="between 1 and 10"):
        service.recommend("test", top_k=11)


def test_service_exposes_catalogue_item() -> None:
    service = RecommendationService(engine=FakeEngine())

    perfume = service.get_perfume("2")

    assert perfume is not None
    assert perfume["name"] == "Candidate Two"
    assert service.get_perfume("missing") is None
