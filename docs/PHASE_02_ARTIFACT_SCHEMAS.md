# Phase 2 artifact schemas

## Reproducibility rules

All Parquet tables are sorted by their declared canonical key and written with PyArrow 25.0.1,
Parquet 2.6, ZSTD level 9, dictionary encoding disabled, statistics enabled, data-page version 1.0,
and 65,536-row groups. Each completed table has a physical file SHA-256 and a semantic SHA-256
over newline-delimited canonical JSON logical records in declared column order.

## Chunk manifest

Path: `artifacts/data/techqa_chunk_manifest.parquet` (large, ignored; checksums/summary tracked).
Canonical key: `(chunk_id)`.

Columns are the `CHUNK_FIELDS` tuple in `retrieval/chunking.py`: stable identifiers, official
document provenance, raw and normalized source hashes, exclusive tokenizer/character offsets,
source-exact text/hash, tokenizer/configuration identity, and corpus/configuration hashes.

## Ranked hits

Path: `artifacts/results/retrieval_ranked_hits.parquet` (Git-trackable research-critical output;
physical and semantic hashes are also tracked). Canonical key:
`(question_id, retrieval_strategy, rank)`. There are ten rows per question and strategy. Hybrid rows
retain both constituent rank and score where present.

## Retrieval conditions

Path: `artifacts/results/retrieval_query_level.parquet` (Git-trackable research-critical output;
physical and semantic hashes are also tracked). Canonical key:
`(question_id, retrieval_strategy, k)`. Ordered lists are canonical JSON. The four depths are
prefixes of one top-10 ranking. Impossible or unresolved rows have null document-metric values and
are excluded from metric denominators. No `y_suff` field exists.

## Development metrics

Path: `artifacts/results/phase02_metrics_train_validation.csv`. Canonical key:
`(split, retrieval_strategy, k)`. Only train and validation are permitted. Each row declares the
eligible denominator, document Recall@k, and MRR@10.

**A correct-document retrieval hit is not proof that the retrieved chunks contain sufficient
answer evidence.** These fields do not measure context sufficiency, answerability, grounding, or
hallucination reduction.

## Manifests and validation

- `phase02_dependency_manifest.json`: exact interpreter/packages, resolved model/tokenizer commit,
  device, dtype, dimension, and smoke outputs.
- `phase02_chunk_summary.json`: counts and length distributions (written after full validation).
- `phase02_chunk_checksums.json`: chunk cache key plus physical/semantic hashes.
- `phase02_integrity_report.json`: row counts, keys, rank continuity, prefix nesting, and test seal.
- `phase02_artifact_hashes.json`: physical and semantic hashes for completed artifacts.
- `phase02_run_manifest.json`: Phase 1 hash/Git SHA, configuration/model identity, cache state,
  timings, cache sizes, row counts, and explicit Phase 3 boundary.
- `phase02_feasibility_report.json`: immutable historical snapshot of measured evidence and the
  approved checkpoint-resume decision made before the full run completed.
