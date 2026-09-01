# Phase 5 Research Decisions

## Governance boundary

Phase 5 starts from frozen Phase 4 commit `c5178a20946729c03faaba07c6754e9a11002d56` and tag `phase-04-answerability-model-v1`. It creates independent governance and does not amend any frozen Phase 1-4 or Phase 14 decision artifact. The authoritative machine-readable method is `configs/phase05_generation_grounding.json`.

The phase keeps three constructs separate: global benchmark question answerability, frozen retrieval-conditioned context sufficiency (`y_suff_final`), and grounding of claims in a generated answer against the exact prompt-visible context. Phase 5 never relabels one construct from another.

## Decision P5-01: population and seal

The primary population is exactly the 89 frozen PRIMARY TechQA VALIDATION questions. Only hybrid retrieval at k=5 and k=10 is generated, for 178 maximum states. TRAIN is not a primary TechQA generation population. TechQA TEST cannot be read by a Phase 5 prompt, generation, quality, grounding, policy-view, paired-comparison, or summary path. RAGTruth TEST is a separate supporting evaluator-validation split and is explicitly permitted only for that purpose.

## Decision P5-02: generator and prompt

The sole generator is `Qwen/Qwen2.5-1.5B-Instruct` at immutable revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; tokenizer revision is identical. It runs on CPU in float32 without quantization, in evaluation/no-gradient mode, with seed 42, greedy one-beam decoding, 128 maximum new tokens, and repetition penalty 1.0. Sampling temperature, top-p, and top-k are absent. No model fallback is allowed.

The exact user prompt is the LF-normalized file `artifacts/governance/phase05_generation_prompt.txt`, SHA-256 `2db430ca27b5dec058b00815d07b21bcf9924f17b0d12006c1a07963ca17e8d1`. It is rendered through the pinned tokenizer's chat template. Benchmark answers, labels, probabilities, retrieval scores, gold evidence, annotations, NLI data, and policy actions are prohibited inputs. An infrastructure-only synthetic smoke test passed before this freeze; it used no TechQA content.

## Decision P5-03: context assembly

Retrieved chunks remain in frozen rank order and are rendered with `[CHUNK n]` separators. The model configuration exposes 32,768 positions; 128 are reserved for output, leaving a maximum chat-templated input of 32,640 tokens. Full chunks are included whenever possible. If truncation is necessary, every earlier chunk remains complete and only the last included chunk may be prefix-token-truncated. No later chunk may leapfrog an earlier chunk. Prompt-visible context and provenance receive stable hashes.

Interpretation stops if k10 fails to expose additional prompt-visible context beyond k5 for at least 20% of questions. This is a pre-results feasibility rule, not an outcome-dependent threshold.

## Decision P5-04: answer quality

Benchmark answers become accessible only after the generation artifact is complete. ROUGE-L is the deterministic F1 from `rouge-score==0.1.2` without stemming. BERTScore reports F1 from `bert-score==0.3.13` using `distilroberta-base` revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, layer 5, English, no IDF, no baseline rescaling, fast tokenizer, float32 CPU inference. These are reference-similarity/correctness measures, not grounding measures.

## Decision P5-05: claims and grounding proxy

Generated responses use deterministic sentence, line, and bullet segmentation with source offsets. No LLM creates or rewrites claims. Non-empty segments containing an alphanumeric character are evaluation claims; zero-claim responses remain explicitly undefined for claim-rate metrics.

For every claim, the frozen Phase 2 MiniLM model ranks the retrieved chunks by cosine similarity. The top three chunks are evaluated independently by the frozen Phase 3 NLI model `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Maximum entailment is the continuous support score; one minus that score is unsupportedness, not hallucination probability. Claim text is never truncated. Premises use the compatible frozen 256-token head-and-tail budget rule. Claims that cannot fit are unevaluable and remain outside rate denominators.

## Decision P5-06: RAGTruth validation and threshold

The actual QA schema contains deterministically alignable human spans, so claim-level validation is used. Official source-grouped TRAIN/TEST partitions remain intact. The primary analysis uses `quality == good`; all-quality results are a declared sensitivity. Human spans marked `implicit_true` remain unsupported relative to supplied context and `due_to_null` remains explicit provenance.

The support threshold grid is 0.00 through 1.00 in 0.02 steps. RAGTruth TRAIN alone selects the threshold by unsupported-class F1, then unsupported precision, then higher threshold. The frozen value is applied once to RAGTruth TEST and then mechanically to TechQA VALIDATION. Source-level 1,000-replicate percentile bootstrap intervals use seed 42. If good-quality RAGTruth TEST response AUROC is below 0.60, binary TechQA grounding results are exploratory; the evaluator is not retuned or replaced.

## Decision P5-07: policies and missing outcomes

Generation occurs once for k5 and once for k10. G0 selects k5; G1 selects k10; G2 uses the frozen 10% Phase 4 trajectory (`t_low=0.78`, `t_high=0.82`); G3 uses the already frozen 20% trajectory (`t_low=0.56`, `t_high=0.72`). Phase 5 does not recompute or optimize thresholds. Abstentions, generation failures, empty outputs, and mathematically undefined claim metrics are NA rather than zero. Coverage and quality remain separate.

## Decision P5-08: stopping and interpretation

The pre-results configuration and governance files become immutable when `phase05_pre_results_governance_freeze.json` is written. Execution failure of a frozen component requires a stop and decision request, never substitution. Empirical interpretation, if produced, lives only in `docs/PHASE_05_RESULTS_INTERPRETATION.md`. Phase 6 inferential testing is not part of this phase.
