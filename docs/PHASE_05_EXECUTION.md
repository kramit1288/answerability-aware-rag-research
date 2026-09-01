# Phase 5 Execution Protocol

## Scope

Phase 5 generates deterministic answers for frozen TechQA VALIDATION hybrid k5/k10 contexts, measures reference-answer quality independently of context grounding, validates the grounding proxy against official RAGTruth QA partitions, applies the frozen proxy to TechQA, and materializes frozen policy and paired views for later Phase 6 analysis. It does not unseal TechQA TEST, retrain Phase 4, alter thresholds, replace the primary policy, or run Phase 6 statistics.

## Stage gates

1. Verify the authoritative branch/commit and all specified Phase 3/4 hashes, manifest entries, populations, and checkers. A scientific hash mismatch stops the phase.
2. Resolve exact model/tokenizer revisions, install and smoke-test the frozen metric implementations, and run one synthetic-only generator smoke test.
3. Freeze the prompt, configuration, research decisions, execution protocol, and artifact schemas in `phase05_pre_results_governance_freeze.json` before assembling or generating any TechQA prompt.
4. Audit RAGTruth schema, offsets, official QA census, and source isolation. Build claim/chunk inputs and compute grounding scores for official TRAIN and TEST without using TEST labels in threshold selection.
5. Select `t_support` on RAGTruth TRAIN only, persist the full grid, freeze the selected threshold, then evaluate RAGTruth TEST with source bootstrap uncertainty.
6. Materialize exactly 178 TechQA VALIDATION hybrid states, verify k5 prefixes and k10 additional context, and stop if the predeclared utilization rule fails.
7. Generate or resume exact k5/k10 states. Cache keys include every frozen revision/config/prompt/context hash. A completed valid state is never recomputed.
8. Close generation artifacts before loading benchmark answers. Compute ROUGE-L and BERTScore separately from grounding.
9. Segment TechQA responses, score grounding with the RAGTruth-frozen evaluator, retain frozen `y_suff_final` only as evaluation metadata, and report unevaluable claims.
10. Construct G0/G1/G2/G3 by selecting existing states. Preserve NA for abstentions and failures. Persist paired k5/k10 records without inferential testing.
11. Add Phase-5-scoped LF rules, build the Phase 5 manifest/integrity report, run all checkers/tests/compile/diff checks, and stop for review without commit or tag.

## Determinism and resumption

All random sources use seed 42. Generator, tokenizer, prompts, contexts, raw outputs, normalized outputs, metric settings, and evaluator scores are content-addressed. JSON uses sorted UTF-8 serialization with trailing LF; CSV uses stable columns and LF; research Parquet uses the frozen deterministic writer settings. Infrastructure retry is bounded to one retry with identical inputs/settings and cannot depend on output quality.

## Failure rules

`generated`, `empty_output`, `generation_failed`, and `no_evaluable_claim` are distinct statuses. Failed generation is never an empty answer. Empty/whitespace output and zero-evaluable-claim results are counted explicitly. Undefined reference and grounding metrics are NA. A frozen model or metric compatibility failure, material k10 context collapse, upstream drift, RAGTruth split leakage, or TechQA TEST access causes a stop rather than a silent workaround.

## Intended commands

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe scripts/run_phase05_generation_grounding.py --config configs/phase05_generation_grounding.json
.venv/Scripts/python.exe scripts/check_phase05.py
.venv/Scripts/python.exe -m pytest --basetemp=.pytest-local-temp
.venv/Scripts/python.exe -m compileall src scripts tests
git -c safe.directory=C:/answerability-aware-rag-research diff --check
```

The local pytest temporary directory is removed after testing. No command in this protocol evaluates TechQA TEST.
