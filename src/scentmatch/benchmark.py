"""Reproducible CPU latency and memory benchmark for ScentMatch."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from scentmatch import __version__
from scentmatch.data.audit import read_jsonl
from scentmatch.service import RecommendationService

DEFAULT_QUERIES = Path("data/eval/ranking_queries_v3.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark_v1")


def percentile(values: list[float], quantile: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 2),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "p99_ms": round(percentile(values, 0.99), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


class MemorySampler:
    """Sample process RSS while the benchmark is running."""

    def __init__(self, process: Any, interval_seconds: float = 0.01) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self.peak_rss = int(process.memory_info().rss)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))


def _megabytes(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def _current_process(psutil: Any) -> Any:
    """Resolve the process even in a container with mismatched PID namespaces."""
    try:
        return psutil.Process(os.getpid())
    except psutil.NoSuchProcess:
        proc_stat = Path("/proc/self/stat")
        if not proc_stat.exists():
            raise
        host_pid = int(proc_stat.read_text(encoding="utf-8").split(maxsplit=1)[0])
        return psutil.Process(host_pid)


def _grouped_latency(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(float(record["latency_ms"]))
    return {
        name: summarize_latencies(values) for name, values in sorted(grouped.items())
    }


def run_benchmark(
    args: argparse.Namespace,
    service_factory: Any = RecommendationService,
) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            'Benchmark dependency is missing. Install: pip install -e ".[app]"'
        ) from error

    queries = read_jsonl(args.queries.resolve())
    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        raise ValueError("Benchmark query set is empty")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")

    process = _current_process(psutil)
    rss_before = int(process.memory_info().rss)
    load_started = time.perf_counter()
    service = service_factory()
    load_ms = (time.perf_counter() - load_started) * 1000.0
    rss_loaded = int(process.memory_info().rss)

    warmup_queries = queries[: max(0, args.warmup)]
    for row in warmup_queries:
        service.recommend(str(row["query_text"]), top_k=args.top_k)

    workload = [dict(row) for _ in range(args.repeats) for row in queries]
    random.Random(args.seed).shuffle(workload)
    records: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    with MemorySampler(process) as sampler:
        for row in workload:
            started = time.perf_counter()
            response = service.recommend(str(row["query_text"]), top_k=args.top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            records.append(
                {
                    "query_id": str(row.get("query_id", "unknown")),
                    "category": str(row.get("category", "unknown")),
                    "reference": response.get("reference") is not None
                    and bool(response["reference"].get("accepted", False)),
                    "latency_ms": round(latency_ms, 4),
                    "results": len(response.get("results", [])),
                }
            )
    benchmark_seconds = time.perf_counter() - benchmark_started
    rss_after = int(process.memory_info().rss)
    total_ram = int(psutil.virtual_memory().total)
    latencies = [float(record["latency_ms"]) for record in records]

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scentmatch_version": __version__,
        "scope": "in-process warm recommendation inference",
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "total_ram_mb": _megabytes(total_ram),
            "device": "cpu",
        },
        "configuration": {
            "query_file": str(args.queries),
            "unique_queries": len(queries),
            "repeats": args.repeats,
            "warmup_queries": len(warmup_queries),
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "startup": {
            "service_load_ms": round(load_ms, 2),
            "rss_before_load_mb": _megabytes(rss_before),
            "rss_after_load_mb": _megabytes(rss_loaded),
            "rss_load_delta_mb": _megabytes(max(0, rss_loaded - rss_before)),
        },
        "inference": {
            "overall": summarize_latencies(latencies),
            "by_category": _grouped_latency(records, "category"),
            "by_reference": _grouped_latency(records, "reference"),
            "total_seconds": round(benchmark_seconds, 3),
            "throughput_queries_per_second": round(
                len(records) / max(benchmark_seconds, 1e-9), 3
            ),
        },
        "memory": {
            "rss_after_benchmark_mb": _megabytes(rss_after),
            "peak_rss_mb": _megabytes(sampler.peak_rss),
            "peak_over_loaded_mb": _megabytes(
                max(0, sampler.peak_rss - rss_loaded)
            ),
        },
        "model": service.info(),
        "notes": [
            "Warm inference excludes service/model loading time.",
            "Latency is measured around RecommendationService.recommend, not HTTP.",
            "Compare results only when hardware, query set, repeats and top_k match.",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    system = report["system"]
    config = report["configuration"]
    startup = report["startup"]
    overall = report["inference"]["overall"]
    memory = report["memory"]
    lines = [
        "# ScentMatch CPU benchmark",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Platform | {system['platform']} |",
        f"| Python | {system['python']} |",
        f"| Logical CPUs | {system['logical_cpu_count']} |",
        f"| Total RAM | {system['total_ram_mb']} MB |",
        "| Device | CPU |",
        "",
        "## Workload",
        "",
        f"- Unique queries: {config['unique_queries']}",
        f"- Repeats: {config['repeats']}",
        f"- Warm-up queries: {config['warmup_queries']}",
        f"- `top_k`: {config['top_k']}",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Service load | {startup['service_load_ms']} ms |",
        f"| Warm p50 | {overall['p50_ms']} ms |",
        f"| Warm p95 | {overall['p95_ms']} ms |",
        f"| Warm p99 | {overall['p99_ms']} ms |",
        f"| Mean | {overall['mean_ms']} ms |",
        (
            "| Throughput | "
            f"{report['inference']['throughput_queries_per_second']} query/s |"
        ),
        f"| Loaded RSS | {startup['rss_after_load_mb']} MB |",
        f"| Peak RSS | {memory['peak_rss_mb']} MB |",
        "",
        (
            "Warm inference excludes service/model loading. Latency covers the "
            "shared `RecommendationService`, not HTTP transport."
        ),
        "",
    ]
    by_reference = report["inference"].get("by_reference", {})
    if by_reference:
        lines.extend(
            [
                "## Latency by request type",
                "",
                "| Request type | Count | p50 | p95 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for value, label in (("False", "No reference"), ("True", "Reference")):
            summary = by_reference.get(value)
            if summary:
                lines.append(
                    f"| {label} | {summary['count']} | {summary['p50_ms']} ms | "
                    f"{summary['p95_ms']} ms |"
                )
        lines.append("")

    by_category = report["inference"].get("by_category", {})
    if by_category:
        lines.extend(
            [
                "## Latency by category",
                "",
                "| Category | Count | p50 | p95 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for category, summary in by_category.items():
            lines.append(
                f"| {category} | {summary['count']} | {summary['p50_ms']} ms | "
                f"{summary['p95_ms']} ms |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(args)
    print(render_markdown(report))
    print(f"JSON: {args.output_dir / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
