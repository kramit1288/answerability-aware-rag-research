# Phase 5 Execution-Only Optimization Amendment

This append-only amendment supplements the frozen Phase 5 pre-results
governance. It does not rewrite or replace `PHASE_05_RESEARCH_DECISIONS.md`,
`PHASE_05_EXECUTION.md`, or the canonical Phase 5 configuration.

## Trigger and scope

The original RAGTruth execution used the frozen CPU float32 DeBERTa NLI
evaluator with `nli_batch_size=16`, two PyTorch compute threads and two
inter-op threads. The host exposes four logical processors. At the time of
amendment, 4,992 of the 42,296 exact RAGTruth claims were durably checkpointed.
Exhaustive inference had unacceptable wall-clock runtime. This amendment was
initiated solely for execution throughput; no partial AUROC, AUPRC, precision,
recall, F1, score distribution, or human-label-conditioned result was read or
used to choose it.

Only execution-equivalent changes are permitted. The model and tokenizer
revisions, float32 semantics, exact response/claim population, segmentation,
candidate selection (`N=3`), premise/hypothesis construction, token budget,
unevaluable handling, source grouping, split separation, threshold procedure,
TechQA outputs, Phase 3/4 state, and TechQA TEST seal remain unchanged.

## Implementation audit

The environment is `torch 2.13.0+cpu` and `transformers 5.15.0`, using CPU.
The current implementation performs true tokenizer batching and one model
forward per batch, uses `torch.inference_mode()`, and uses dynamic longest-in-
batch padding (not fixed maximum-length padding). Completed checkpoint claim IDs
are filtered before tokenization and NLI inference. Length bucketing is not
used.

## Equivalence benchmark

The benchmark used the first 16 checkpointed claims in deterministic claim-ID
order (48 already persisted candidate pairs). It read no labels and calculated
no partial scientific-performance metric. Batch sizes 16, 32, and 64 were
tested with one, two, and four compute threads; the original 16/2/2 execution
was the reference. Every candidate had zero NLI argmax-class changes, zero
probability differences above `1e-6` and `1e-5`, and maximum absolute
difference below `1e-5`. Full details are persisted in
`artifacts/results/phase05_execution_benchmark.json`.

## Frozen selected execution

The fastest numerically equivalent candidate was selected by claims/second:

- device: CPU
- dtype: float32
- NLI batch size: 32 (execution-only; the frozen scientific configuration and
  exact pair population remain unchanged)
- `torch.set_num_threads(4)`
- `torch.set_num_interop_threads(1)`
- dynamic longest-in-batch padding: enabled (unchanged)
- input-length-aware bucketing: enabled; original pair order is restored before
  checkpoint persistence
- `torch.inference_mode()`: enabled (unchanged)

Benchmark throughput was 0.501870 claims/second for the original reference and
0.684124 claims/second for the selected execution, a 1.363149x speedup. The
selected execution is implemented as a runtime-only amendment and resumes from
the existing checkpoint. Completed claim IDs are not recomputed.

The amendment is represented canonically by
`artifacts/results/phase05_execution_optimization.json`, whose configuration
identity is the frozen Phase 5 canonical SHA-256
`b35ca88eccd8a24194a2976a12b31f82cc9ea243856c8c379001a84250972dd3`.
The amendment canonical SHA-256 is
`c7eb0b7e80d811803022eedf8ff26ac2fa98134d2e0722a2ccc37f58790a9fc7` and its
physical LF-byte SHA-256 is
`48133fa4b887ab002ce6f92d60711c653d047f3d0f6e22192227a59bc0627e68`.

## Resume and safeguards

The RAGTruth stage requires this amendment, verifies its identity against the
frozen configuration, applies only the selected CPU thread settings, and then
filters the existing append-only checkpoint before inference. No RAGTruth
sampling is introduced. TechQA TEST remains sealed, and Phase 6 remains
unstarted.
