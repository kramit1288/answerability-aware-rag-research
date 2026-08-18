# Phase 2 decision request — controlled retrieval parameters

## Status - resolved 2026-08-18

The researcher approved the seven decisions on 2026-08-18, including the pre-experiment tokenizer-
bounded chunking amendment. The binding choices are now recorded in
`docs/EXPERIMENT_CONTRACT.md` and `docs/RESEARCH_DECISIONS.md`. This request is retained as an
audit trail of the alternatives considered; it no longer blocks Phase 2 execution.

Before approval, Phase 2 full execution was blocked before corpus chunking, indexing, model
loading, or ranking.
The Phase 1 dependency passed its standalone checker and the frozen semantic split SHA-256 is
`4e84c5bdce2d6649f7e9eb8a22271faf0ffb51a5c934fbb642721237cb4292e7`.

The experiment contract fixes the corpus, chunk size/overlap, dense model/revision, normalized-dot
similarity, equal-weight RRF constant, depths, and chunk-ID tie-breaking. It does not define all
methodological parameters that the Phase 2 implementation request explicitly requires to be frozen.
Choosing them inside code would silently add result-changing decisions after the contract freeze.

No test retrieval ranking or aggregate retrieval metric has been calculated or inspected.

## Missing decisions

### 1. BM25 tokenizer and preprocessing

The contract refers to a “declared technical-token tokenizer” but never declares its algorithm.
The prototype uses lowercasing plus the regex `[a-z0-9_./:+#-]+`; the audit says that tokenizer
must be reimplemented only after it is versioned and tested.

Feasible alternatives:

1. Freeze the prototype-compatible tokenizer: Unicode NFKC, casefold, then
   `[a-z0-9_./:+#-]+`, identically for chunks and queries.
2. Use a Unicode word tokenizer with punctuation splitting.
3. Use the dense model tokenizer for BM25 terms.

Scientific consequences: token boundaries materially change term frequencies, document lengths,
technical identifier preservation, and rankings. Options 2 and 3 also reduce continuity with the
technical-token behavior already audited from the prototype.

Recommendation: option 1. It preserves error codes, paths, versions, `C++`/`C#`, underscores,
slashes, colons, plus signs, dots, and hyphens as technical tokens while making normalization
explicit and deterministic.

### 2. BM25 implementation, variant, and parameters

The contract says BM25 but does not specify the library/variant, `k1`, `b`, negative-IDF handling,
or any other implementation-specific parameter. `pyproject.toml` mentions `rank-bm25` only through
an open lower bound; that is not a frozen method.

Feasible alternatives:

1. `rank_bm25.BM25Okapi` with explicit `k1=1.5`, `b=0.75`, and `epsilon=0.25`.
2. A sparse-matrix BM25 implementation with equivalent-looking parameters.
3. A custom BM25 implementation with fully specified equations and IDF clipping.

Scientific consequences: BM25 variants and IDF treatment can change scores and ordering,
particularly for frequent technical terms. Implementations that appear equivalent are not
guaranteed to produce identical rankings.

Recommendation: option 1, explicitly pinned. It matches the prototype’s declared library and
current project dependency while making its previously implicit defaults part of the contract.

### 3. Chunk-boundary and offset serialization behavior

The contract fixes 360 whitespace-delimited words, overlap 80, inclusive/exclusive word offsets,
and a conceptual chunk-ID formula. It does not fully specify:

- whether the final partial chunk is retained and whether a redundant overlap-only tail is emitted;
- exact whitespace token boundary behavior after the Phase 1 cleaning transform;
- how the concatenated chunk-ID inputs are serialized unambiguously;
- normalized character offsets and their relationship to raw source character offsets; or
- provenance mapping when NFKC, whitespace collapse, or data-URI removal prevents exact raw
  substring reconstruction.

Feasible alternatives:

1. Use normalized-document tokens from Python `re.finditer(r"\S+")`, step 280, retain one final
   partial chunk, stop when `end_word == document_word_count`, use exclusive word/character ends,
   and hash a canonical JSON tuple of ID inputs. Persist normalized character offsets plus a
   derived word-to-raw-span provenance map.
2. Use `str.split()` and word offsets only, with no raw/normalized character mapping.
3. Chunk raw text first and normalize each chunk afterward.

Scientific consequences: options 2 and 3 weaken or change source alignment. Option 3 can create
different boundaries from the normalized corpus shared by retrieval.

Recommendation: option 1. It gives deterministic boundaries and audit-preserving offsets while
making explicit that normalized chunk text is not always an exact raw substring.

### 4. Dense query/passage encoding behavior

The model and commit are fixed, but the contract does not say whether queries or passages use
prompts/prefixes, whether original or normalized question text is encoded, what maximum-length
truncation behavior applies, or whether query and passage batch settings share one encode path.

Feasible alternatives:

1. Encode original question strings and normalized chunk strings directly with the same
   `SentenceTransformer.encode` path, no prompts or prefixes, model-default tokenizer truncation,
   float32 outputs, and `normalize_embeddings=True`.
2. Add `query:`/`passage:` prefixes.
3. Use model-specific prompt names if exposed by a later Sentence Transformers version.

Scientific consequences: prefixes and prompt templates can materially change embeddings and
ranking. Model-default truncation must also be recorded because 360 whitespace words may exceed
the MiniLM tokenizer limit.

Recommendation: option 1, with the resolved tokenizer/model configuration and maximum sequence
length persisted. It matches the frozen model’s conventional symmetric usage without introducing
an unregistered prompt intervention.

### 5. Dense execution dtype/device and exact-search backend

The contract requires normalized embeddings and dot product but only says hardware is recorded.
It does not freeze computation dtype/device or the exact-search implementation. Small numeric
differences can change near ties before the stable chunk-ID tie-break is applied.

Feasible alternatives:

1. CPU float32 encoding plus blocked NumPy float32 matrix multiplication and deterministic
   lexicographic ordering by `(-score, chunk_id)`.
2. GPU float32 encoding/search with deterministic PyTorch settings.
3. GPU float16 encoding/search.

Scientific consequences: options 2 and 3 are faster but hardware/kernel dependent; float16 can
alter ranking. ANN is excluded by the contract and is not an alternative.

Recommendation: option 1 for the primary frozen run, subject to a measured smoke/runtime check.
If it is materially infeasible, record evidence and request a pre-test resource decision rather
than silently switching device or dtype.

### 6. RRF constituent ranking depth / candidate pool

The contract fixes equal weights and `RRF k=60` but does not say whether ranks are computed over the
full chunk corpus or a truncated candidate union. This is explicitly required by the Phase 2 gate.

Feasible alternatives:

1. Compute exact full-corpus BM25 and dense ranks, then RRF every chunk and select the hybrid
   top 10.
2. Fuse the union of each retriever’s top 1,000 candidates.
3. Fuse only each retriever’s top 10.

Scientific consequences: truncated pools can omit chunks that would enter the fused top 10 and
make RRF depend on an additional arbitrary depth. Option 3 is especially restrictive.

Recommendation: option 1. It best matches the contract’s definition of complete deterministic
rankings `R_(i,r)` and avoids candidate-pool depth as an uncontrolled factor.

### 7. Serialization implementation for the contract’s Parquet chunk manifest

The contract names `artifacts/data/techqa_chunk_manifest.parquet`, but no Parquet engine,
compression, row-group behavior, or exact dependency is selected. No serialization package is
currently installed.

Feasible alternatives:

1. Pin PyArrow and a deterministic Parquet schema/compression configuration.
2. Use an ignored canonical JSONL materialization plus a lightweight tracked CSV checksum manifest.
3. Use pandas with whichever Parquet backend is present.

Scientific consequences: this should not change rankings if semantic hashes are canonicalized
independently of file bytes, but it changes interoperability, cache validation, and byte-level
reproducibility. Option 3 is environment-dependent.

Recommendation: option 1 for the contract path, while defining the chunk-corpus semantic hash over
canonical record content rather than Parquet bytes.

## Dependency feasibility observed before the stop

The repository virtual environment currently has CPython 3.14.3, NumPy 2.5.2, and SciPy 1.18.0.
It does not currently contain pandas, PyTorch, Transformers, Sentence Transformers, `rank-bm25`,
an exact-search package, PyArrow, tokenizers, safetensors, or Hugging Face Hub. The pinned dense
model is not present in the inspected local Hugging Face cache.

This is not yet recorded as an incompatibility: the complete stack was deliberately not installed
or smoke-tested after the methodological gate failed. After the seven choices above are approved,
Phase 2 should first create an exact-version environment manifest/lock and run the requested model,
normalization, exact-search, and BM25 smoke test. Any Python 3.14 wheel/model interoperability
failure must stop the full run with exact evidence.

## Requested approval

Approve or amend the seven recommendations above. After approval, add the result-changing choices
to `docs/RESEARCH_DECISIONS.md` before producing any retrieval ranking, then implement and execute
Phase 2. Until then, the test set and all retrieval outcomes remain sealed.
