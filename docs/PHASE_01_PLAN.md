# Phase 1 Plan — Data Foundation Only

## 1. Objective and boundary

Phase 1 will establish a pinned, validated, leakage-resistant data foundation for TechQA-RAG-Eval
and RAGTruth. It will download and preserve raw releases, validate real schemas, assign stable IDs
and source groups, create and persist the TechQA train/validation/test split, respect RAGTruth’s
official split, explicitly flag reference-evidence anomalies, freeze the validated TechQA
component/split artifacts, and emit reproducibility manifests.

Phase 1 will **not**:

- chunk or index the corpus;
- implement BM25, dense, or hybrid retrieval;
- create context-sufficiency labels;
- manually align evidence or decide the final evidence eligibility of the two empty-reference
  answerable rows;
- extract model features;
- fit/calibrate a classifier;
- tune thresholds;
- generate answers;
- inspect test predictions; or
- produce experimental result tables.

Chunking is specified in `docs/EXPERIMENT_CONTRACT.md` but begins in Phase 2. Phase 1 records
document-level inputs and stable IDs so Phase 2 can generate chunks reproducibly. It records
`DEV_Q014` and `DEV_Q094` as `reference_status=empty_answerable` with
`gold_evidence_analysis_status=pending_manual_alignment`; it does not invent or infer spans.

## 2. Pinned inputs

### TechQA-RAG-Eval

- repository: `nvidia/TechQA-RAG-Eval`
- config/split: `default/train`
- revision: `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`
- `train.json` SHA-256:
  `69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7`
- `corpus.zip` SHA-256:
  `c06aa287dcc1abf8db6b49b8495df095db73342d729f6451ac330785245d10be`

### RAGTruth

- repository: `ParticleMedia/RAGTruth`
- revision: `c103204b9ce28d6bbad859304bf30de72b8ed8fe`
- `response.jsonl` SHA-256:
  `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`
- `source_info.jsonl` SHA-256:
  `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`

Downloads must target immutable revision URLs. A checksum mismatch stops the run; the script must
not update expected hashes or accept a changed `main` branch silently.

## 3. Files to create or change in Phase 1

The following is the planned change set, not work completed in Phase 0.

### Configuration and project metadata

| Path | Action | Purpose |
|---|---|---|
| `configs/phase01_data.json` | Create | Pinned URLs/revisions/hashes, raw/derived paths, split seed/ratios, connected-component rules, optimization objective/version, freeze policy, and schema version. JSON avoids introducing a YAML dependency. |
| `.gitignore` | Create or update | Exclude large raw/derived data and transient downloads while retaining manifests and `.gitkeep` files. |
| `pyproject.toml` | Change only if required | Add any direct runtime dependency actually imported by Phase 1 and register pytest markers such as `network`. Do not add modelling dependencies for future phases. |
| `README.md` | Update | Document the two Phase 1 commands, raw-data policy, and test-set seal. |
| `docs/RESEARCH_DECISIONS.md` | Change only if required | The approved grouped split is already recorded in Phase 0. Add only an approved implementation deviation or newly resolved methodological choice; do not rewrite the decision silently. |

### Package modules

| Path | Action | Responsibility |
|---|---|---|
| `src/answerability_rag/config.py` | Create | Typed loading/validation for `phase01_data.json`; canonical config hash. |
| `src/answerability_rag/hashing.py` | Create | Streaming SHA-256, canonical JSON hashing, normalized text/question hashes. |
| `src/answerability_rag/io.py` | Create | Atomic downloads/writes, immutable-file checks, safe ZIP extraction, canonical CSV/JSON serialization. |
| `src/answerability_rag/data/__init__.py` | Create | Public data-foundation API only. |
| `src/answerability_rag/data/schemas.py` | Create | Typed records/dataclasses and required-field/type validators. |
| `src/answerability_rag/data/techqa.py` | Create | Load/validate QA rows, corpus archive, document IDs, context-to-corpus alignment, and corpus manifest. |
| `src/answerability_rag/data/ragtruth.py` | Create | Load/validate sources/responses, join on `source_id`, enforce official split, and create response manifest. |
| `src/answerability_rag/data/splits.py` | Create | Duplicate/source graph, connected components, deterministic constrained component assignment, objective diagnostics, freeze hashes, and overlap checks. |
| `src/answerability_rag/data/manifests.py` | Create | Build the four required CSV manifests and dataset/reproducibility metadata. |
| `src/answerability_rag/reproducibility.py` | Create | Python/package/platform capture and seed/config metadata; no model runtime logic. |

### Thin scripts

| Path | Action | Purpose |
|---|---|---|
| `scripts/prepare_phase01_data.py` | Create | Orchestrate download/verification, extraction, validation, grouping/splitting, and manifest writes. Contains no scientific logic beyond calling package functions. |
| `scripts/check_phase01_data.py` | Create | Read persisted outputs and run all integrity checks without regenerating or altering assignments. |

### Tests and fixtures

| Path | Action | Purpose |
|---|---|---|
| `tests/conftest.py` | Create | Small local fixtures; tests do not depend on live network by default. |
| `tests/fixtures/techqa_tiny.json` | Create | Synthetic schema fixture covering answerable/impossible, shared document, and duplicate question cases. It must be clearly synthetic and never used as research data. |
| `tests/fixtures/techqa_corpus_tiny.zip` | Create | Tiny safe ZIP fixture including nested-path and duplicate-content edge cases. Generate with a documented test helper if binary review is inconvenient. |
| `tests/fixtures/ragtruth_source_tiny.jsonl` | Create | Synthetic source rows with QA/non-QA task types. |
| `tests/fixtures/ragtruth_response_tiny.jsonl` | Create | Synthetic six-response/source rows, official splits, label fields, and quality states. |
| `tests/test_hashing.py` | Create | Stable streaming/canonical hashes and normalization behaviour. |
| `tests/test_safe_io.py` | Create | Atomic writes, checksum failure, rerun immutability, and ZIP path-traversal rejection. |
| `tests/test_techqa_schema.py` | Create | Required fields/types, ID uniqueness, context structure, null/empty handling, and attached-context alignment. |
| `tests/test_corpus_manifest.py` | Create | Stable document IDs, unique paths, raw hashes, safe extraction, and no prefix-only deduplication. |
| `tests/test_group_splits.py` | Create | Connected-component grouping, deterministic optimal assignment, zero source/question overlap, ratio/class diagnostics, provenance-prefix non-use, freeze/reuse, and split inheritance. |
| `tests/test_ragtruth_schema.py` | Create | Source/response keys, unique IDs, six-response grouping, label schema, and official split consistency. |
| `tests/test_manifests.py` | Create | Exact columns/dtypes, row counts, canonical ordering, null policy, and byte-identical reruns. |
| `tests/test_phase01_integration.py` | Create | Optional network/cache integration test against the pinned releases, marked `network`. |

No existing notebook or `notebooks/original_prototype.py` will be changed.

## 4. Function and interface plan

Function names may change only for a documented engineering reason; their contracts may not be
weakened silently.

### 4.1 Configuration and immutable I/O

`load_phase01_config(path: Path) -> Phase01Config`

- validates revision strings, 64-character hashes, split ratios summing to 1, seed, and paths;
- returns immutable typed configuration; and
- computes a canonical configuration SHA-256.

`download_verified(url: str, destination: Path, expected_sha256: str) -> FileRecord`

- streams to a same-directory temporary file;
- verifies checksum before atomic rename;
- refuses to overwrite a different existing file;
- permits an idempotent rerun when the existing checksum matches; and
- records URL, resolved URL, bytes, hash, and retrieval time.

`safe_extract_zip(archive: Path, destination: Path) -> list[ExtractedFile]`

- rejects absolute paths, `..`, device paths, links, and destinations escaping the declared root;
- never overwrites a non-identical extracted file;
- preserves archive-relative paths; and
- hashes extracted bytes while reading.

`write_csv_atomic(...)` and `write_json_atomic(...)`

- use canonical column/key order, UTF-8, explicit null encoding, and atomic rename;
- return output hash and row count; and
- never append to an existing research artifact.

### 4.2 TechQA loading and schema validation

`load_techqa_rows(path: Path) -> list[TechQAQuestion]`

- parses the pinned top-level JSON list;
- performs no silent coercion of malformed context values; and
- preserves raw strings exactly while computing separate normalized representations.

`validate_techqa_rows(rows) -> ValidationReport`

For the pinned revision it checks:

- exactly 910 rows and exactly the five expected fields;
- unique non-empty string IDs/questions;
- Boolean `is_impossible`;
- string answers and list contexts;
- context dictionaries with exactly `filename` and `text` strings;
- 610 non-impossible and 300 impossible rows;
- exactly one non-empty context for every non-impossible row and none for impossible rows;
- answer `"-"` for every impossible row;
- exactly `DEV_Q014` and `DEV_Q094` are non-impossible rows with empty answers; both are recorded
  as `reference_status=empty_answerable` and
  `gold_evidence_analysis_status=pending_manual_alignment`, not repaired; and
- provenance prefixes are recognized but not interpreted as final splits.

Hard-coded counts are appropriate because the revision and hashes are pinned. A different release
requires an explicit contract/decision update rather than adapting silently.

`build_corpus_manifest(archive, extracted_root) -> DataFrame`

- creates `doc_id = "techqa-doc:" + archive_relative_path`;
- requires 28,481 unique non-directory paths in the pinned archive;
- records filename separately from full path, sizes, raw SHA-256, and strict UTF-8 status;
- applies the frozen normalization/cleaning transform in memory to record `normalized_sha256`,
  normalized character count, cleaning version, removed-payload count, and status, without writing
  a normalized corpus or retrieval chunks;
- detects duplicate basenames and duplicate content without deleting either; and
- uses complete hashes, never the first 500 characters, for identity checks.

`validate_context_corpus_alignment(rows, corpus_manifest, extracted_root)`

- requires every attached gold filename to resolve uniquely in the corpus;
- requires every attached context text to byte/text-match that official document for this pinned
  revision;
- reports the number of linked and unlinked official documents; and
- does not build a retrieval corpus from linked contexts.

### 4.3 Stable grouping and split assignment

`normalize_question_for_grouping(text: str) -> str`

- Unicode NFKC;
- casefold;
- punctuation to spaces; and
- collapse/trim whitespace.

The normalized text is used only through SHA-256 in the manifest. The original question remains
available in raw data.

`build_question_components(rows) -> ComponentBuildResult`

- builds an undirected graph linking rows with the same gold filename or normalized-question
  hash;
- computes connected components deterministically;
- assigns `split_group_id = "techqa-group:" + sha256(component_signature)`, where the canonical
  signature contains sorted normalized-question hashes and sorted gold filenames but no
  `TRAIN_Q`/`DEV_Q` provenance value;
- ensures duplicate questions and shared gold documents transitively remain together; and
- returns both question-to-component membership and component rows containing members, shared
  filenames, normalized-question hashes, component size, and answerability counts.

`assign_grouped_splits(rows, components, ratios=(0.70,0.15,0.15), seed=42) -> SplitAssignmentResult`

- assigns whole connected components with a deterministic constrained optimizer;
- first minimizes the maximum normalized absolute deviation across the target total and both
  answerability-class counts for each split, then minimizes their summed normalized absolute
  deviation while holding the first objective at its optimum;
- requires an optimal solver status; a non-optimal, infeasible, or time-limited result stops the
  phase rather than silently accepting a heuristic allocation;
- resolves equivalent optimal assignments using a seed-42, provenance-neutral component-signature
  tie order and
  records the solver, solver version, objective values, and tie rule;
- never splits a component;
- uses original `TRAIN_Q`/`DEV_Q` only as provenance, not assignment; and
- emits exactly one assignment for every question.

`validate_split_assignments(rows, assignments)`

- checks unique/full question coverage;
- verifies no `split_group_id`, gold filename, or normalized-question hash crosses splits;
- checks exact rerun stability;
- checks each split contains both global classes;
- independently recomputes total/class deviations and verifies the persisted optimum and declared
  approximately 70/15/15 objective;
- checks that changing or removing the derived `provenance_partition` values does not change component
  membership or assignment;
- reports exact achieved ratios and class prevalence without imposing an arbitrary tolerance or
  manually moving questions; and
- computes and records the component-manifest and assignment-file hashes that later phases must
  consume.

`freeze_split_artifacts(component_manifest, assignments, metadata)`

- writes both files atomically only after every split-integrity check passes;
- stores their SHA-256 values and freeze timestamp in `dataset_metadata.json`;
- permits an idempotent rerun only when regenerated semantic content and hashes match; and
- rejects later replacement or reassignment unless an explicit pre-test research decision changes
  the contract and all dependent artifacts are regenerated.

The test assignment may be counted/hashed for these checks. No retrieval/model output is allowed
in Phase 1.

### 4.4 RAGTruth loading and official split enforcement

`load_ragtruth_sources(path) -> list[RAGTruthSource]`

`load_ragtruth_responses(path) -> list[RAGTruthResponse]`

- stream JSONL with line-numbered parse errors;
- retain all task types and qualities in raw/manifests; and
- do not normalize away span offsets.

`validate_ragtruth(sources, responses) -> ValidationReport`

For the pinned revision it checks:

- 2,965 unique sources and 17,790 unique responses;
- required source/response field sets and types;
- exactly six responses per `source_id`;
- every response joins exactly one source and no source is orphaned;
- official split values are only `train`/`test` and each `source_id` has exactly one split;
- official counts of 15,090/2,700 responses;
- QA counts of 989 sources and 5,934 responses;
- QA split counts of 839/150 sources and 5,034/900 responses;
- QA `source_info` has string `question` and `passages`; and
- every label has integer offsets, string text/type, Boolean `implicit_true` and `due_to_null`,
  and offsets/text consistent with the response. Any known annotation exception is reported
  explicitly rather than silently corrected.

`build_ragtruth_manifest(sources, responses) -> DataFrame`

- emits one response row joined to source metadata;
- retains official split and `source_id`;
- hashes response, prompt, and canonical source info;
- records quality, label count/types, `implicit_true`, and `due_to_null`; and
- marks `primary_grounding_eligible = (task_type == "QA" and quality == "good")` without
  discarding other rows.

### 4.5 Metadata and integrity report

`capture_phase01_metadata(config, inputs, outputs, validations) -> dict`

records:

- run/config/schema versions and timestamps;
- Python, package, OS, and command information;
- master seed and split algorithm version;
- source URLs, repository revisions, input hashes/bytes;
- observed row/schema/count summaries;
- TechQA reference-status counts and the two pending manual-alignment IDs;
- connected-component counts, achieved split/class ratios, optimizer status/objectives/version,
  component-manifest hash, split-assignment hash, and freeze timestamp;
- every produced artifact path/hash/row count;
- every check and pass/fail status; and
- deviations or warnings.

It does not claim model reproducibility because no model exists in Phase 1.

## 5. Execution sequence

1. Validate configuration and output roots.
2. Download pinned TechQA README/JSON/corpus archive and RAGTruth JSONL files into revision-named
   raw directories, verifying hashes before accepting them.
3. Safely extract the official TechQA corpus to a revision-named derived directory without
   changing file contents.
4. Parse and validate TechQA rows and official corpus.
5. Verify attached contexts against official corpus documents.
6. Create stable document/question IDs and connected split components.
7. Solve the approved deterministic grouped 70/15/15 assignment, validate all overlap/objective
   rules, and stage the component and assignment manifests for freezing.
8. Parse and validate all RAGTruth sources/responses and official source-grouped split.
9. Write the four required CSV manifests and `dataset_metadata.json` atomically; record and freeze
   the validated component/split hashes and freeze timestamp.
10. Run `check_phase01_data.py` solely from persisted files.
11. Run unit tests and the pinned integration test.
12. Review the integrity report and stop. Do not start chunking/retrieval.

## 6. Commands to run

From repository root in a clean virtual environment:

```powershell
python -m pip install -e ".[dev]"
python scripts/prepare_phase01_data.py --config configs/phase01_data.json
python scripts/check_phase01_data.py --config configs/phase01_data.json
python -m pytest -q
python -m pytest -q -m network tests/test_phase01_integration.py
```

If the network test is skipped due to a documented offline environment, preparation must use a
previously verified raw cache with the same hashes; fixture-only tests are not sufficient to call
the milestone complete.

No notebook command is part of Phase 1.

## 7. Generated raw, derived, and artifact files

### Immutable raw inputs

```text
data/raw/techqa/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a/
  README.md
  train.json
  corpus.zip
data/raw/ragtruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/
  response.jsonl
  source_info.jsonl
```

### Derived but non-experimental data

```text
data/derived/techqa/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a/corpus/
  ...exactly extracted official documents...
```

Extraction is reproducible from `corpus.zip`; it does not replace the raw archive. No normalized
or chunked retrieval text files are produced until Phase 2; Phase 1 records the deterministic
normalization hashes and diagnostics required by the corpus-manifest schema.

### Required Phase 1 artifacts

`artifacts/data/techqa_split_components.csv`

- exact schema from the experiment contract;
- one row per connected component, including component members, linking filenames/question hashes,
  class counts, assigned split, and component hash;
- canonical sort by `split_group_id`.

`artifacts/data/techqa_split_assignments.csv`

- exact schema from the experiment contract;
- one row per 910 question;
- includes `split_group_id`, frozen split, `provenance_partition`, `reference_status`, and
  `gold_evidence_analysis_status`; and
- canonical sort by `question_id`.

`artifacts/data/techqa_corpus_manifest.csv`

- exact document-level schema from the experiment contract;
- one row per 28,481 archive document;
- canonical sort by `archive_path`.

`artifacts/data/ragtruth_manifest.csv`

- exact response-level schema from the experiment contract;
- one row per 17,790 response;
- canonical sort by numeric/string-stable `response_id`.

`artifacts/data/dataset_metadata.json`

- source/input/output hashes, schemas, counts, environment, commands, split diagnostics, and
  validation outcomes.

`artifacts/data/phase01_integrity_report.json`

- machine-readable check ID, description, expected, observed, status, severity, and details;
- overall status is pass only if every P0 integrity check passes.

Phase 1 creates nothing under `artifacts/results/`, `artifacts/models/`, or
`artifacts/figures/`.

## 8. Test plan

### Deterministic unit tests

- normalization is stable for Unicode, punctuation, whitespace, and empty strings;
- SHA-256 functions stream bytes and canonical JSON deterministically;
- existing raw file with a matching hash is accepted; mismatch fails without overwrite;
- interrupted download cannot appear at final path;
- ZIP traversal/absolute paths are rejected;
- valid TechQA rows parse without type coercion;
- malformed/missing fields, duplicate IDs, wrong context types, and impossible/context
  contradictions fail with row IDs;
- exactly the two pinned empty-answerable IDs receive the pending manual-alignment flags and no
  evidence location;
- document IDs use full paths and do not deduplicate by prefix or partial text;
- duplicate-content documents are retained and reported;
- shared-document and duplicate-question relationships form transitive components;
- component records contain all and only their question members and deterministic diagnostic links;
- all rows in a component receive one split;
- same config/seed/input yields byte-identical assignments;
- different iteration order yields the same assignments;
- the constrained optimizer reports an optimum and persisted objective values recompute exactly;
- changing/removing the derived `provenance_partition` value does not change grouping or
  assignment;
- no gold filename/question hash/group crosses splits;
- a frozen assignment is reused when hashes match and a differing reassignment is rejected;
- all RAGTruth responses for a source keep the released split;
- cross-split `source_id`, orphan join, wrong response multiplicity, or malformed label offsets fail;
- manifests have exact columns/order/types and canonical row order; and
- metadata lists every input/output hash and failed checks prevent overall pass.

### Pinned integration checks

- TechQA file/revision hashes match;
- observed TechQA row/class/context/corpus counts match the contract;
- the two known empty-answerable reference IDs and their pending status match the contract;
- all 610 attached contexts resolve and exactly match official corpus files;
- component/split overlap checks pass, the assignment optimizer reports an optimum, and frozen
  component/assignment hashes are present;
- RAGTruth file/revision hashes and full/QA split counts match;
- the preparation command is idempotent and a second run changes no accepted artifact bytes; and
- the standalone checker passes using persisted artifacts alone.

## 9. Acceptance criteria

Phase 1 is complete only when all of the following are true:

1. The raw files are stored under immutable revision paths with the expected SHA-256 hashes.
2. The observed TechQA and RAGTruth schemas/counts match the pinned contract or the phase stops
   for an explicit contract review.
3. The TechQA corpus manifest contains all 28,481 official documents; no query-derived corpus is
   substituted.
4. Every attached TechQA context resolves to its exact official corpus document.
5. All 910 TechQA question IDs and all 17,790 RAGTruth response IDs are unique and represented in
   their manifests.
6. `DEV_Q014` and `DEV_Q094` are explicitly flagged as answerable rows with empty references and
   pending manual alignment; no evidence span is fabricated.
7. Every TechQA question belongs to exactly one deterministic connected component and split, and
   the component manifest fully accounts for all 910 questions.
8. Gold filename, normalized duplicate question, and connected-component overlap are exactly zero
   across TechQA train/validation/test; `TRAIN_Q`/`DEV_Q` remain provenance only.
9. The constrained assignment has an optimal solver status, its objective recomputes exactly, and
   the achieved approximately 70/15/15 sizes and class balance are recorded without manual moves.
10. RAGTruth’s official split is preserved and no `source_id` crosses train/test.
11. All required artifacts match the experiment-contract schemas and include hashes/row counts.
12. The component and assignment manifests are frozen after validation; their hashes and freeze
    timestamp are persisted and required as immutable inputs by later phases.
13. Re-running preparation with the same inputs/config produces byte-identical split/manifests or
    explicitly stable semantic hashes where timestamps are intentionally excluded from content.
14. Unit and pinned integration tests pass; any skip/failure is reported and prevents completion
    unless the user approves a documented deviation.
15. The phase completion report lists files changed, exact commands, passed/failed checks,
    generated artifacts/hashes, unresolved issues, and deviations.
16. No retrieval, sufficiency-label, manual evidence alignment, model, calibration, threshold,
    generation, or final-test experiment code has been implemented or run.

## 10. Approved constraints and implementation-contingent observations

The following choices are approved and are not open for data-dependent revision in Phase 1:

1. Use the connected-component 70/15/15 TechQA research split; `TRAIN_Q`/`DEV_Q` are provenance
   fields only.
2. Build components from shared gold/source filename and normalized duplicate-question links;
   impossible rows can only be grouped by duplicate question because no gold source is supplied.
3. Retain the full 28,481-document official corpus across all question splits. The split protects
   learned labels/models, not access to the fixed documentation collection.
4. Treat the current hashes/revisions as frozen inputs; any upstream change triggers a new
   decision rather than automatic schema adaptation.
5. Freeze and reuse the component and split assignments after validation.
6. Flag the two known empty-answerable references without aligning evidence during Phase 1.

The actual connected-component distribution, attainable split ratios/class balance, and optimizer
objective values cannot be known until Phase 1 executes. They are implementation outputs, not an
invitation to revise the approved method after viewing results. An unexpected giant component,
schema mismatch, checksum mismatch, or inability to prove an optimal assignment stops Phase 1 for
review.

After Phase 1 acceptance, stop and request review before Phase 2.
