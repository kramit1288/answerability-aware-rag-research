# Phase 5 Execution Decision Request

## Status

Phase 5 generation and answer-quality stages completed. The frozen RAGTruth
grounding evaluator is executable, but the required CPU-only execution is not
completing within the available runtime.

## Evidence

- TechQA generation: 178/178 validation states completed successfully.
- Answer quality: 178/178 states scored with the frozen ROUGE-L and BERTScore
  configuration.
- RAGTruth grounding: 4,992 of 42,296 claims were scored and persisted in the
  append-only resumable checkpoint.
- Runtime: the mandated DeBERTa NLI evaluator uses CPU float32 and frozen
  `nli_batch_size = 16`; observed throughput projects to many additional hours
  on this four-logical-processor host.
- No model, precision, token-budget, candidate-selection, threshold, or
  policy substitution has been made.

## Decision required

Please choose one of the following before Phase 5 grounding evaluation is
resumed:

1. Authorize a documented execution-only optimization that does not change the
   evaluator model or scientific definitions (for example, fixed CPU thread
   parallelism or deterministic sharding), with a new reproducibility record;
   or
2. Provide an execution environment/runtime in which the frozen configuration
   can complete unchanged; or
3. Direct that Phase 5 stop with the partial RAGTruth checkpoint retained.

The current checkpoint is resumable and no TechQA TEST data has been opened.
No Phase 5 post-results interpretation or final artifact manifest has been
created, and Phase 6 has not started.
