# Modeling and ranking

This document describes the main retrieval, fine-tuning, community-ranking and link-prediction components used by the current ScentMatch system.

## Dense retrieval baseline

The initial semantic baseline used `intfloat/multilingual-e5-small` with normalized embeddings and cosine similarity implemented as a NumPy matrix product. Documents use the `passage:` prefix and user requests use `query:`.

For 5,000 catalogue items, a dedicated vector database was unnecessary for the baseline. Embedding-to-catalogue consistency is protected with SHA-256 checks.

The original 40-query silver-label evaluation produced:

| Metric | Baseline |
| --- | ---: |
| Precision@5 | 0.310 |
| Precision@10 | 0.265 |
| HitRate@5 | 0.750 |
| HitRate@10 | 0.850 |
| MRR@10 | 0.501 |
| nDCG@10 | 0.198 |

These are proxy metrics against automatically generated relevance labels, not human preference scores.

## Fine-tuned multilingual E5

`intfloat/multilingual-e5-small` was fine-tuned on synthetic bilingual retrieval examples with explicit hard negatives. Evaluation queries were kept outside training.

On the 20-query test split, the recorded comparison was:

| Metric | Baseline | Fine-tuned | Fine-tuned + exclusions |
| --- | ---: | ---: | ---: |
| Precision@5 | 0.310 | 0.530 | **0.600** |
| Precision@10 | 0.270 | 0.510 | **0.560** |
| HitRate@5 | 0.700 | 0.850 | **0.950** |
| HitRate@10 | 0.800 | 0.900 | **0.950** |
| MRR@10 | 0.525 | 0.658 | **0.783** |
| nDCG@10 | 0.217 | 0.368 | **0.412** |

The fine-tuned model improved nDCG@10 on 30 of 40 queries. After adding deterministic handling of explicit exclusions such as `без ванили` / `without vanilla`, 36 queries improved and 4 regressed relative to the baseline.

Occasion-oriented requests remained the weakest category, which is consistent with the training data being primarily grounded in notes, accords and exclusions.

## Structured constraints and hybrid inference

Dense similarity is not trusted to enforce hard negation. ScentMatch therefore extracts only explicit forbidden notes/accords from phrases such as `без ...`, `исключить ...`, `no ...` and `without ...` and filters violating candidates from an expanded retrieval pool.

Other attributes remain ranking signals rather than hard-coded global rules.

## Community-aware ranking

The current ranker combines independent evidence sources instead of treating one community field as ground truth:

1. fine-tuned multilingual E5 similarity;
2. structured intent and hard constraints;
3. season, day/night, longevity, sillage and perceived-gender votes;
4. vote-backed `reminds_me_of` similarity;
5. `also_liked` affinity;
6. Bayesian rating and popularity signals;
7. diversity control.

Weights are query-dependent. A free-form request starts primarily from semantic retrieval and quality signals and activates additional axes only when they are requested. A `similar to X` query additionally activates reference-content and graph signals.

Season evidence is smoothed with a global prior. When a fragrance has at least 30 seasonal votes and fewer than 10% support the explicitly requested season, it may be excluded. This is a conservative guardrail rather than a claim that seasonality is objective.

Brand-aware MMR-style diversification is applied after ranking to reduce redundant top-k results.

## Reference resolution

Named perfume references are resolved conservatively with typo tolerance, confidence/margin checks and abstention for ambiguous cases. Latin names can appear inside either Russian or English requests. Single-word names require stronger contextual evidence to prevent ordinary query terms from being mistaken for a perfume name.

The resolved reference activates an additional reference-aware scoring path and the reference itself is removed from the result list.

## Community similarity link prediction

For resolved reference queries, ScentMatch uses an offline link-prediction model built from observed vote-backed `reminds_me_of` pairs.

Labels:

- positive: at least 10 votes and agreement >= 0.70;
- negative: at least 10 votes and agreement <= 0.40;
- uncertain pairs are discarded;
- unobserved pairs are never treated as negatives.

The final model uses features including:

- fine-tuned E5 cosine similarity;
- note/accord overlap;
- accord-intensity cosine similarity;
- season and day/night similarity;
- longevity, sillage and rating similarity;
- popularity percentiles;
- `also_liked` evidence;
- common-neighbor Jaccard and Adamic-Adar computed only from the train positive graph.

The portable model is `StandardScaler + LogisticRegression` with `C=0.1` and balanced class weights. Parameters are serialized to JSON so runtime inference does not depend on pickle/joblib compatibility.

Node degree was deliberately removed after development ablation exposed hub bias. The feature decision was made before final test inspection.

Final test performance is summarized in [`evaluation.md`](evaluation.md).

## Evidence-based explanations

The application does not use a generative LLM to invent explanations. Each recommendation is explained only from available scoring evidence. The output can include:

- `why_recommended` typed reasons;
- vote-backed `possible_drawbacks`;
- an evidence level (`high`, `medium`, `limited`);
- raw score components and active weights for audit.

A reason is emitted only if the underlying signal exists. Community summaries are retained only with vote evidence and remain explicitly labelled as community-derived information.

## Retired experiments

Several intermediate branches were useful during development but are not part of the final public architecture:

- an early Aromo-based community augmentation had incomplete coverage and was replaced by the later community pipeline;
- a Persian review-chunk reranker was prototyped, but semantic similarity could confuse statements such as “too heavy for summer” with positive summer relevance because polarity was not modeled explicitly;
- manually changing ranker weights from a handful of examples was rejected because it risks selection bias.

These experiments informed the current design but their version-by-version documents are intentionally not kept in the public documentation set.
