# Phase 2 controlled-retrieval execution

## Commands

From the repository root with the canonical virtual environment:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python scripts\check_phase01_data.py
.\.venv\Scripts\python scripts\smoke_phase02_dependencies.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts\run_phase02_retrieval.py
.\.venv\Scripts\python scripts\check_phase02_retrieval.py
```

The run command validates the frozen Phase 1 hash/counts, uses one chunk corpus for all retrievers,
rejects stale caches, resumes only completed dense batches, writes the two query-level tables, and
aggregates train/validation only. Repeating it after completion must reuse the content-addressed
caches and reproduce semantic and physical table hashes.

## Current execution status

Phase 2 completed under run ID `phase02-e57ea98d4538acfb`. The dependency/model smoke test passed,
and the frozen Phase 1 split hash was verified. One corpus of 169,472 chunks represents all 28,481
official documents; no eligible non-empty document produced zero chunks. The canonical CPU
checkpoint completed all 169,472 float32, normalized MiniLM embeddings and was validated for shape,
cache key, finite values, and unit norms (`0.99999982` to `1.00000012`).

The run produced 27,300 top-10 ranked-hit rows and 10,920 nested condition rows. The artifact checker
passes all nine checks, including rank continuity, unique chunks, nested k prefixes, expected split
counts, and absence of aggregate test rows. Aggregate document-level metrics exist only for train and
validation in `artifacts/results/phase02_metrics_train_validation.csv`. A correct-document retrieval
hit is not proof that the retrieved chunks contain sufficient answer evidence.

The first invocation that completed the interrupted checkpoint and retrieval took 14,320.53 seconds:
9,325.66 seconds for the final embedding segment, 4,905.27 seconds for BM25 queries, and 53.05 seconds
for dense queries. Earlier embedding work was safely committed over prior checkpointed invocations,
so there is intentionally no invented single uninterrupted cold-run timer. The immediately repeated
cache-only run took 4,546.26 seconds, including 4,456.81 seconds for BM25 queries and 52.22 seconds for
dense queries. It reproduced both semantic and physical hashes for the ranked-hit and condition
tables exactly.

No aggregate test retrieval performance was calculated or reported, and no Phase 3 logic was
implemented.
