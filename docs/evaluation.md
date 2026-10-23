# Evaluation and metric provenance

This document is the canonical public evaluation summary for ScentMatch. It separates regression checks, leakage-safe graph evaluation and proxy retrieval metrics from subjective recommendation quality.

## Evaluation principles

ScentMatch uses different evaluations for different failure modes instead of collapsing everything into one score:

- query-intent extraction and reference resolution are tested as deterministic regression components;
- semantic retrieval is compared with fixed silver-label query sets;
- reference similarity is evaluated as link prediction on observed vote-backed graph pairs;
- CPU latency/RAM is measured separately because it depends on hardware;
- unrestricted subjective recommendation quality is **not** claimed to be solved without a larger human-judged set.

The original retrieval protocol fixed queries and relevance rules before running the first embedding baseline. Explicit forbidden conditions receive priority in relevance judgments. Candidate pools should combine multiple retrieval systems rather than being sourced from one baseline only.

## Query understanding

Source: the v2 regression evaluation set.

| Metric | Value | Cases |
| --- | ---: | ---: |
| Intent case accuracy | 1.000 | 18 |
| Intent field accuracy | 1.000 | 18 |
| Reference Top-1 accuracy | 1.000 | 15 positive |
| Reference abstention accuracy | 1.000 | 5 negative/ambiguous |

These cases are curated engineering regression tests. They demonstrate that the implemented examples remain stable, not that every possible natural-language formulation is understood.

## Community similarity link prediction

Canonical model artifact: `artifacts/link_predictor_v1/model.json`.

Labels use only observed pairs:

- positive: >= 10 votes and agreement >= 0.70;
- negative: >= 10 votes and agreement <= 0.40;
- missing/unobserved pairs are not negatives.

The deterministic split is made on canonical undirected pairs, so `(A, B)` and `(B, A)` cannot leak across train/dev/test.

| Metric | Dev | Test |
| --- | ---: | ---: |
| ROC-AUC | 0.840 | **0.883** |
| Average Precision | 0.756 | **0.790** |
| E5 macro pairwise accuracy | 0.803 | 0.709 |
| ML macro pairwise accuracy | **0.947** | **0.894** |
| E5 top-1 positive rate | 0.800 | 0.688 |
| ML top-1 positive rate | **0.938** | **0.896** |

The test classification set contains 893 observed pairs, including 200 positive pairs. Pairwise ranking covers 96 reference groups and 158 positive-vs-negative comparisons.

### Leakage control

- directed records are canonicalized to one undirected pair before splitting;
- dev/test positive edges are removed from graph features such as common neighbors;
- split selection is deterministic;
- node degree was removed before final test inspection after development ablation exposed hub bias;
- final test metrics were not used to choose regularization or features.

## Community coverage

The selected catalogue has complete community-profile coverage in the final pipeline:

| Statistic | Value |
| --- | ---: |
| Selected catalogue | 5,000 |
| Matched community profiles | 5,000 (100%) |
| `reminds_me_of` edges in catalogue | 27,804 |
| `also_liked` edges in catalogue | 70,956 |
| Profiles with vote-backed pros | 4,199 |
| Profiles with vote-backed cons | 3,324 |

## Proxy retrieval metrics

Early semantic-retrieval experiments use automatically generated silver relevance labels. These scores are useful for controlled model-to-model comparisons but can reward the same structured rules used to generate the labels.

The final fine-tuned + explicit-exclusion test run reported:

| Metric | Result |
| --- | ---: |
| Precision@5 | 0.600 |
| Precision@10 | 0.560 |
| HitRate@5 | 0.950 |
| HitRate@10 | 0.950 |
| MRR@10 | 0.783 |
| nDCG@10 | 0.412 |

For that reason the main README promotes the leakage-safe link-prediction metrics more prominently than the silver retrieval scores.

## Blind human-evaluation design

A later evaluation design was prepared to compare multiple ranking variants without exposing system identity to the annotator. The intended pool contains 48 bilingual queries across season, occasion, performance, exclusions, attributes, popularity, reference, complex and negative-control categories.

Each `query × perfume` pair is intended to receive:

- relevance 0–3;
- explicit-constraint violation flag;
- evidence-quality score 0–2;
- `unknown` when the evaluator cannot judge the fragrance reliably.

System names, scores and source ranks are hidden during annotation. Parameters should be changed only on development queries; test queries remain sealed until the model is fixed.

This design is documented as a **future/partial evaluation plan**, not as a completed human benchmark. A larger expert-judged set and online feedback are still required before making product-level claims about subjective recommendation quality.

## What is not measured yet

- expert-judged NDCG for unrestricted free-form recommendations;
- online click/save/purchase feedback;
- individual preference calibration;
- live retailer availability or current-price quality;
- robust generalization across user populations.

Hardware performance is reported separately in [`benchmarking.md`](benchmarking.md).
