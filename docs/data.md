# Data and dataset design

This document summarizes the dataset choices, catalogue construction, profile schema and training-data preparation used by ScentMatch.

## Sources and licensing

The primary catalogue is based on the Kaggle **Fragrantica.com Fragrance Dataset** published by Olga G (Miufana1):

- source page: https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset
- stated origin: web scraping of Fragrantica.com;
- stated licence: CC BY-NC-SA 4.0;
- verified archive files: `fra_cleaned.csv` and `fra_perfumes.csv`.

The source files are not committed or redistributed with the repository. ScentMatch keeps code, checksums and small derived evaluation artifacts only. The project is presented as a non-commercial research/portfolio project.

Community signals used by the current ranking pipeline come from the Le Decanteur Fragrantica Perfumes dataset referenced in the main README. Community summaries are treated as evidence only when backed by votes; they are not presented as objective perfume properties.

## Raw catalogue structure

`fra_cleaned.csv` contains 24,063 rows with structured perfume metadata, including brand/name, gender, year, rating statistics, top/middle/base notes, perfumers and up to five main accords.

`fra_perfumes.csv` contains 70,103 rows with descriptions, ratings, accords, perfumers and URLs. The two files are joined by the numeric Fragrantica ID extracted from the URL rather than the full URL string. All 24,063 structured profiles had a matching description using this ID.

## Selecting the 5,000-item catalogue

The public runtime catalogue is a deterministic subset intended to stay representative while remaining practical on a CPU-only laptop.

Selection rules:

1. join the two raw files by numeric Fragrantica ID;
2. deduplicate normalized `brand + name + year`, preferring higher rating count, then rating and description length;
3. require a complete note pyramid, description length of at least 80 characters and at least 100 ratings;
4. allocate proportional quotas by `gender + primary_accord`;
5. rank within each group by rating count, rating and description length;
6. cap each brand at 50 fragrances and refill unused quota from remaining candidates.

For the verified source snapshot this produced:

- 24,063 linked profiles;
- 23,973 profiles after deduplication;
- 13,680 eligible candidates;
- exactly 5,000 selected fragrances;
- 455 brands;
- no more than 50 fragrances from one brand;
- median rating count: 882.5 (minimum 220).

## Data-quality audit

The 5,000 generated profiles were checked automatically. The verified audit reported:

- 5,000 profiles and 5,000 unique `perfume_id` values;
- no empty note layers or accord lists;
- no empty or duplicated `retrieval_text` values;
- no placeholder terms such as `unknown`, `null` or `n/a` inside term lists;
- no HTML, URLs or obvious encoding corruption in descriptions;
- no case-only duplicates among normalized terms;
- maximum `retrieval_text` length of 1,073 characters and approximately 877 characters at the 99th percentile.

Exact duplicates inside one note layer are removed deterministically while preserving the first occurrence. Similar-but-not-identical terms are not aggressively merged because that could remove useful semantic distinctions.

## `perfume_profile` schema

Each catalogue item is converted to a single structured document. The machine-readable schema lives in `schemas/perfume_profile.schema.json`.

```json
{
  "schema_version": "1.0",
  "perfume_id": "74630",
  "name": "Example",
  "brand": "Example Brand",
  "source_url": "https://www.fragrantica.com/perfume/...-74630.html",
  "gender": "unisex",
  "country": "France",
  "year": 2022,
  "rating": {"value": 4.12, "count": 381},
  "perfumers": ["Perfumer Name"],
  "accords": ["woody", "amber", "warm spicy"],
  "notes": {
    "top": ["Bergamot"],
    "middle": ["Rose"],
    "base": ["Sandalwood"]
  },
  "source_description": "Source description...",
  "retrieval_text": "Example by Example Brand. ..."
}
```

The original note pyramid and accord ordering are preserved. `source_description` remains separate from the deterministic `retrieval_text` so provenance stays explicit.

## Fine-tuning data

The first fine-tuning dataset contains:

- 2,000 train queries;
- 200 validation queries;
- 2,200 distinct positive perfumes;
- three hard negatives per query (6,600 total);
- four balanced query groups: `note`, `accord`, `combo`, `negation`.

Evaluation queries `q001`–`q040` are excluded from generation and training. Positive IDs do not overlap between train and validation.

Hard negatives are mined with the frozen `intfloat/multilingual-e5-small` baseline from highly similar results that violate an explicit query condition. Mean cosine similarity was 0.7984 for positives and 0.8249 for hard negatives; in 99.4% of mined cases the negative ranked above the positive under the baseline.

The ML-ready files contain only text columns (`anchor`, `positive`, `negative_1..3`); IDs, scores, ranks and mining reasons are kept separately for provenance. The training notebook uses `MultipleNegativesRankingLoss` and `BatchSamplers.NO_DUPLICATES`.

## Important limitations

- The source data comes from scraped community sites; licence claims are those published with the datasets, not by Fragrantica itself.
- Ratings and community votes represent community behaviour, not objective perfume quality.
- The source descriptions are primarily English; Russian support relies on multilingual representations and bilingual query handling.
- The synthetic training set is useful for explicit notes, accords and exclusions, but it is not a substitute for human preference labels.
