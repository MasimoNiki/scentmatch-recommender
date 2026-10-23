# CPU benchmark protocol

## Goal

Measure reproducible application performance on the actual target laptop rather than copying numbers from a cloud runner or a different CPU.

## Command

From the project root with the virtual environment activated:

```powershell
scentmatch-runtime-check
scentmatch-benchmark --repeats 3 --warmup 2 --top-k 5
```

The workload uses all 48 queries from `data/eval/ranking_queries_v3.jsonl`, shuffles 144 measured calls with seed 42 and preserves query categories for grouped reporting.

| Category | Unique queries | What it checks |
| --- | ---: | --- |
| season | 6 | summer, winter, spring and autumn preferences |
| occasion | 6 | office, date, everyday use, wedding and holiday |
| performance | 4 | longevity and sillage |
| exclusion | 4 | excluded notes or accords |
| attributes | 4 | desired notes and accords |
| popularity | 4 | well-known/popular preference |
| reference | 10 | similarity to a named fragrance, including a typo |
| complex | 6 | several simultaneous constraints |
| negative control | 4 | vague or underspecified requests |

Each query is measured three times, producing 144 sequential calls. Two calls warm the already loaded service before the measured workload. The benchmark does not call an external LLM.

This is a performance benchmark only. Ranking quality and model metrics are documented in [`evaluation.md`](evaluation.md).

## Measured values

The benchmark records:

- service/model load time;
- RSS before and after model load;
- warm inference mean, p50, p95 and p99;
- throughput in queries per second;
- latency by category;
- reference versus non-reference latency;
- peak process RSS sampled every 10 ms;
- OS, Python, logical CPU count and total RAM.

The timed region is `RecommendationService.recommend`: query encoding, intent/reference resolution, ranking, optional link prediction, diversity and explanation generation. HTTP transport is excluded.

## Fair-run checklist

1. Close model-heavy applications and previous ScentMatch processes.
2. Keep the laptop connected to power and use the normal performance profile.
3. Do not run under an IDE debugger.
4. Run the benchmark once rather than selecting the best result from several runs.
5. Preserve the generated JSON without manually editing values.
6. Record whether the run is native Windows or Docker and do not mix the two.

## Output

```text
artifacts/benchmark_v1/benchmark_report.json
artifacts/benchmark_v1/benchmark_report.md
```

The public README reports the preserved native Windows result and explicitly labels latency as warm in-process inference.

## Quick smoke mode

The following command only verifies that the benchmark runner works and must not be reported as the public performance result:

```powershell
scentmatch-benchmark --limit 3 --repeats 1 --warmup 1
```
