import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from scentmatch.api import create_app
from scentmatch.service import RecommendationService
from tests.test_service import FakeEngine


def test_health_and_recommendation_contract() -> None:
    app = create_app(RecommendationService(engine=FakeEngine()))

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/api/v1/recommend",
            json={"query": "Похожий на Reference One", "top_k": 1},
        )

    assert health.status_code == 200
    assert health.json()["catalog_size"] == 2
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "6.0"
    assert payload["results"][0]["explanation"]["why_recommended"]


def test_unknown_perfume_returns_404() -> None:
    app = create_app(RecommendationService(engine=FakeEngine()))

    with TestClient(app) as client:
        response = client.get("/api/v1/perfumes/missing")

    assert response.status_code == 404
