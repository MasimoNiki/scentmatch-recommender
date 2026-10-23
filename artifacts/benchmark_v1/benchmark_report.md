# ScentMatch CPU benchmark

Generated: `2026-08-18T19:02:21.758220+00:00`

## Environment

| Field | Value |
| --- | --- |
| Platform | Windows-10-10.0.19045-SP0 |
| Python | 3.12.4 |
| Logical CPUs | 12 |
| Total RAM | 15706.56 MB |
| Device | CPU |

## Workload

- Unique queries: 48
- Repeats: 3
- Warm-up queries: 2
- `top_k`: 5

## Results

| Metric | Value |
| --- | ---: |
| Service load | 48448.64 ms |
| Warm p50 | 163.68 ms |
| Warm p95 | 830.58 ms |
| Warm p99 | 1189.0 ms |
| Mean | 261.52 ms |
| Throughput | 3.824 query/s |
| Loaded RSS | 873.81 MB |
| Peak RSS | 981.56 MB |

Warm inference excludes service/model loading. Latency covers the shared `RecommendationService`, not HTTP transport.

## Latency by request type

| Request type | Count | p50 | p95 |
| --- | ---: | ---: | ---: |
| No reference | 123 | 159.16 ms | 240.35 ms |
| Reference | 21 | 742.75 ms | 1194.2 ms |

## Latency by category

| Category | Count | p50 | p95 |
| --- | ---: | ---: | ---: |
| attributes | 12 | 168.14 ms | 203.51 ms |
| complex | 18 | 148.39 ms | 191.86 ms |
| exclusion | 12 | 145.05 ms | 174.87 ms |
| negative_control | 12 | 159.3 ms | 277.5 ms |
| occasion | 18 | 181.25 ms | 247.45 ms |
| performance | 12 | 191.2 ms | 265.52 ms |
| popularity | 12 | 148.73 ms | 218.08 ms |
| reference | 30 | 697.82 ms | 1188.76 ms |
| season | 18 | 166.53 ms | 236.55 ms |
