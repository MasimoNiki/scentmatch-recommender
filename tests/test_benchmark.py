from scentmatch.benchmark import percentile, render_markdown, summarize_latencies


def test_percentiles_and_latency_summary() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert percentile(values, 0.5) == 30.0
    summary = summarize_latencies(values)
    assert summary["p50_ms"] == 30.0
    assert summary["p95_ms"] == 48.0


def test_benchmark_markdown_contains_portfolio_metrics() -> None:
    report = {
        "generated_at": "2026-08-18T00:00:00+00:00",
        "system": {
            "platform": "Windows-10",
            "python": "3.11.9",
            "logical_cpu_count": 8,
            "total_ram_mb": 16384.0,
        },
        "configuration": {
            "unique_queries": 48,
            "repeats": 3,
            "warmup_queries": 2,
            "top_k": 5,
        },
        "startup": {"service_load_ms": 1000.0, "rss_after_load_mb": 700.0},
        "inference": {
            "overall": {
                "p50_ms": 80.0,
                "p95_ms": 120.0,
                "p99_ms": 150.0,
                "mean_ms": 90.0,
            },
            "throughput_queries_per_second": 11.1,
        },
        "memory": {"peak_rss_mb": 760.0},
    }

    markdown = render_markdown(report)

    assert "Warm p95" in markdown
    assert "Peak RSS" in markdown
    assert "Windows-10" in markdown
