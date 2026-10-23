from scentmatch.service import RecommendationService
from scentmatch.web import _format_response
from tests.test_service import FakeEngine


def test_frontend_formatter_shows_evidence_and_limitations() -> None:
    service = RecommendationService(engine=FakeEngine())

    markdown = _format_response(
        service.recommend("Похожий на Reference One", top_k=1)
    )

    assert "Candidate Two — Brand B" in markdown
    assert "Почему рекомендован" in markdown
    assert "не объективную истину" in markdown
