"""Gradio demo mounted on the ScentMatch FastAPI application."""

from __future__ import annotations

import argparse
import json
from typing import Any

from scentmatch.api import create_app
from scentmatch.service import RecommendationService

EXAMPLE_QUERIES = [
    "Лучшие мужские ароматы на лето",
    "Популярный несладкий мужской аромат, похожий на Versace Pour Homme",
    "Стойкий свежий аромат на лето без сильного шлейфа",
    "Похожий по настроению на Terre d'Hermes, но для лета",
    "Женский аромат для офиса без ванили",
    "A fresh everyday fragrance with soft sillage",
]


def _format_response(response: dict[str, Any]) -> str:
    lines = ["## Рекомендации"]
    if response.get("query_summary"):
        lines.append("**Распознано:** " + ", ".join(response["query_summary"]))
    reference = response.get("reference")
    if reference and reference.get("display_name"):
        lines.append(f"**Референс:** {reference['display_name']}")
    for warning in response.get("warnings", []):
        lines.append(f"> ⚠️ {warning}")

    for item in response["results"]:
        explanation = item["explanation"]
        lines.extend(
            [
                "",
                f"### {item['rank']}. {item['name']} — {item['brand']}",
                f"Уровень evidence: **{explanation['evidence_level']}**",
            ]
        )
        accords = ", ".join(item.get("accords", [])[:5])
        if accords:
            lines.append(f"Основные аккорды: {accords}")
        if explanation["why_recommended"]:
            lines.append("**Почему рекомендован:**")
            for reason in explanation["why_recommended"]:
                lines.append(f"- {reason['text']}")
        if explanation["possible_drawbacks"]:
            lines.append("**Что стоит учесть (community summary, оригинал):**")
            for drawback in explanation["possible_drawbacks"]:
                lines.append(
                    f"- {drawback['text']} "
                    f"(agreement {drawback['agreement']:.0%}, "
                    f"votes {drawback['votes']})"
                )
        if item.get("source_url"):
            lines.append(f"[Профиль источника]({item['source_url']})")
    lines.extend(
        [
            "",
            "---",
            (
                "Результаты отражают сигналы каталога и коллективные оценки, "
                "а не объективную истину о восприятии аромата."
            ),
            f"Latency после загрузки модели: {response['latency_ms']:.0f} ms",
        ]
    )
    return "\n\n".join(lines)


def build_interface(service: RecommendationService) -> Any:
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError('App dependencies are missing. Install ".[app]".') from error

    def recommend(query: str, top_k: int) -> tuple[str, str]:
        try:
            response = service.recommend(query, int(top_k))
        except ValueError as error:
            return f"⚠️ {error}", "{}"
        return (
            _format_response(response),
            json.dumps(response, ensure_ascii=False, indent=2),
        )

    with gr.Blocks(title="ScentMatch") as demo:
        gr.Markdown(
            "# ScentMatch\n"
            "Подбор ароматов по свободному описанию с прозрачными ML/community "
            "объяснениями."
        )
        with gr.Row():
            query = gr.Textbox(
                label="Опишите желаемый аромат",
                placeholder=(
                    "Например: популярный несладкий мужской аромат на лето, "
                    "похожий на Versace Pour Homme"
                ),
                lines=3,
                scale=5,
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Количество результатов",
                scale=1,
            )
        submit = gr.Button("Подобрать ароматы", variant="primary")
        gr.Examples(EXAMPLE_QUERIES, inputs=query, label="Примеры запросов")
        output = gr.Markdown()
        with gr.Accordion("Технический JSON", open=False):
            raw_output = gr.Code(language="json")
        submit.click(
            recommend,
            inputs=[query, top_k],
            outputs=[output, raw_output],
            api_name=False,
        )
        query.submit(
            recommend,
            inputs=[query, top_k],
            outputs=[output, raw_output],
            api_name=False,
        )
    return demo


def app_main() -> None:
    parser = argparse.ArgumentParser(description="Run ScentMatch API and demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import gradio as gr
        import uvicorn
    except ImportError as error:
        raise RuntimeError('App dependencies are missing. Install ".[app]".') from error

    service = RecommendationService()
    api = create_app(service, ui_path="/app")
    demo = build_interface(service)
    app = gr.mount_gradio_app(api, demo, path="/app")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    app_main()
