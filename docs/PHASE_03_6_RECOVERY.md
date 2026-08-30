# Phase 3.6 interrupted-run recovery

## Phase 3.6c reuse verification

Before the final expanded threshold refinement, all recovered Phase 3.6 caches and Phase 3.6b
artifacts passed their existing integrity checkers. The selected-model development aggregate had
450 persisted condition rows (150 per frozen candidate), the primary semantic label artifact had
6,180 rows, and the active selected-model checkpoints remained complete at 482/482 development
pairs and 15,999/15,999 evaluable primary pairs. Phase 3.6c consumed only the persisted condition
aggregates and did not invoke NLI inference. Historical Phase 3.6b artifact hashes remained
unchanged. TEST was not loaded and Phase 4 was not started.

## Phase 3.6b provenance follow-up

On 2026-08-30 the two annotation files were found to have correct, preserved contents under
reversed provenance paths. They were swapped atomically by filename only. The original 86/64 bytes
now have SHA-256 `04c0f33db4ca3ac9d2a58322f6399c1688ef66a48855bbb7c7546ca7d5851959`
at `phase03_annotation_annotator_1.csv`; the reviewed 85/65 bytes retain SHA-256
`3e7a87f2da694cbb40930c142549cd70dde86c0d89429885711549cf91b99cc4` at the adjudicated path.
No annotation value changed during this filename/provenance correction. All existing Phase 3.6
scores and checkpoints remained valid and were reused without NLI inference. Phase 3.6b then
stopped at its failed development gate; TEST remained sealed and Phase 4 was not started.

Recovery inspection date: 2026-08-29 to 2026-08-30 (Asia/Calcutta)  
Recovery status: complete at the planned pending-human-confirmation checkpoint; Phase 4 not started

## Git state before resuming

- Branch: `phase-03-context-sufficiency`
- Commit: `e6f455751a70936e04e725d877cf65180da12372`
- Staged changes: none
- Modified tracked files: `.gitignore`, `docs/EXPERIMENT_CONTRACT.md`,
  `docs/PHASE_03_ARTIFACT_SCHEMAS.md`, `docs/PHASE_03_EXECUTION.md`, and
  `docs/RESEARCH_DECISIONS.md`
- Untracked Phase 3/3.6 source, config, test, documentation, and result files were present. The
  requested `git status --short`, `git diff`, `git diff --cached`, and five-entry decorated log
  were captured before this note was created.

## Recovered Phase 3.6 state

| Item | State at recovery | Recovery decision |
|---|---|---|
| Frozen semantic config | Present: `configs/phase03_semantic.json`; base canonical SHA-256 recorded as `3f4171a0e5c5edaa6c796f88bc97d7f74c4500c19955eeec09373f56c8f7b6ad` | Preserve; verify frozen inputs before execution. |
| Candidate revisions | Present in config for all three approved candidates | Preserve and verify exact local snapshots. |
| Hugging Face model snapshots | Exact-revision snapshot directories and full weight/tokenizer files are present for all three candidates; no `.incomplete`, `.tmp`, or lock files were found | Preserve; validate by local loading/scoring rather than redownload. |
| Candidate 1 development checkpoint | Current-config checkpoint has 482/482 unique parseable pair rows | Reuse after expected-key validation. |
| Candidate 2 development checkpoint | Current-config checkpoint has 482/482 unique parseable pair rows, but subsequent dtype verification proved the model had loaded its F16 repository weights despite the frozen F32 config | Archive as invalid F16 inference and regenerate in explicitly enforced F32. |
| Candidate 3 development checkpoint | Current-config checkpoint has 224/482 unique parseable pair rows | Resume from the first missing deterministic pair after fixing/validating the rendering-budget implementation. Do not repeat the 224 completed pairs. This completed successfully at 482/482. |
| Superseded development checkpoints | One 4,410-row Candidate-1 cache and one 64-row Candidate-2 cache from the superseded all-single/all-pair formulation remain present | Preserve as excluded historical caches. Their filenames lack the amended base-config hash, so current cache lookup rejects them. |
| Development score Parquets | Present for Candidates 1 and 2 only: 300 condition rows and 1,006 claim rows | Internally complete for those two candidates, but incomplete as final candidate comparison; regenerate derived tables only after Candidate 3 completes. |
| Threshold search CSV | Present with 840 rows: 420 grid rows each for Candidates 1 and 2 | Incomplete as the frozen three-candidate search; regenerate from completed candidate scores. |
| Selected semantic config | Present, selecting Candidate 2, but records Candidate 3 as incompatible after a `balanced premise rendering exceeded frozen max_sequence_length` exception | Not a valid final three-candidate freeze. It was archived after Candidate 3 was recovered and Candidate 2 was proven to have used the wrong dtype. |
| Primary selected-model checkpoint | Present with 208 unique parseable rows, keyed by the provisional semantic-config SHA-256 `98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf` | Subsequent dtype verification proved these Candidate-2 scores used F16 rather than frozen F32. The run reached 272 rows before this was detected; the checkpoint and manifest were stopped and archived as invalid rather than reused or deleted. |
| Final semantic label/claim artifacts | Missing | Generate only after valid three-candidate selection. |
| Strict-vs-semantic final comparison | Missing | Generate only after valid semantic labels. |
| New 100-row confirmation package | Missing | Generate only after semantic model/configuration is frozen. |
| Phase 3.6 checker/tests | Source and unit tests are present | Run after upstream integrity verification and before long inference. |
| Execution timing/logs | No Phase 3.6 stdout/stderr run log was found. Hugging Face/Xet download logs and filesystem timestamps exist. | Report pre-shutdown work only from persisted checkpoints/timestamps; do not reconstruct a single uninterrupted duration. |

All inspected current checkpoints contain parseable JSONL rows, unique pair keys, the expected
model/revision per file, finite probabilities, and probability sums within floating-point tolerance
of one. These are preliminary recovery checks; full expected-key, upstream-hash, dependency,
schema, and row-count validation remains required before reuse.

## Integrity findings during resume

- Phase 1, Phase 2, and Phase 3 integrity checkers passed. The four frozen hashes matched exactly,
  the prototype notebook SHA-256 remained `406a3c0d4781592dcf9824691147c6a271082a368efb0a9c949b8fcf43dd4b73`,
  and the 150-row annotation file remained
  `04c0f33db4ca3ac9d2a58322f6399c1688ef66a48855bbb7c7546ca7d5851959`.
- The BART failure was an implementation budgeting error, not model incompatibility. BART's
  decode/re-tokenize step added tokenizer-specific overhead after the equal head/tail budget was
  calculated. The renderer now deterministically trims only measured excess while retaining the
  frozen 256-token limit and equal constituent budgeting. The recovered BART cache reached 482/482.
- Safetensors inspection proved Candidate 1 and Candidate 3 weights are F32, while Candidate 2's
  repository weights are F16. Transformers 5.15 defaults to the repository/config dtype when no
  explicit `dtype` is passed. The scorer had recorded `float32` without enforcing it. Candidate-2
  scores were therefore inconsistent with the frozen `dtype: float32` execution config and were
  stopped before semantic labels were written.
- The invalid Candidate-2 development and primary checkpoints/manifests were renamed with
  `.invalid-float16` suffixes. Incomplete two-candidate/incorrect-dtype derived outputs were renamed
  with `.pre-float32-recovery` suffixes. Nothing was deleted. Candidate 1 and Candidate 3 raw
  checkpoints remain available for exact reuse.
- The scorer now passes `dtype=torch.float32` explicitly and fails unless all floating model
  parameters are actually F32. Candidate-2 development inference was regenerated, all three
  candidates were compared in F32, and the valid comparison again selected Candidate 2 at exact
  revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. The selected semantic configuration SHA-256 is
  `98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf`.

## Safe-resume blocker on 2026-08-30

- Primary TRAIN+VALIDATION scoring resumed from valid 16-pair checkpoints and reached
  13,744/16,023 unique pairs. The manifest was last atomically updated at
  `2026-08-30T04:52:04.231003+00:00`; completed work remains reusable.
- Scoring then stopped on the frozen long-claim guard. The semantic config explicitly specifies
  `long_claim_policy: fail_if_claim_exceeds_model_pair_budget` and `claim_truncation: false`.
- Exactly two of the 515 eligible TRAIN+VALIDATION questions contain a segmented reference claim
  that cannot fit the selected model's 256-token pair budget: `DEV_Q066` has a 343-token claim and
  `TRAIN_Q526` has a 434-token claim (both counts exclude pair-special tokens).
- Completing Phase 3.6 therefore requires choosing a new rule: truncate reference claims, amend
  deterministic claim segmentation, or exclude affected conditions. Each option changes the
  frozen methodology and cache identity, so no option was chosen during recovery.
- No semantic label artifact or human-confirmation sample was written before this guard fired.

## Approved recovery decision

On 2026-08-30, before semantic label generation, independent confirmation sampling, Phase 4, or
TEST evaluation, the conservative semantic-unevaluable rule was approved. It applies mechanically
to any question for which at least one unchanged frozen-segmentation claim cannot fit the selected
model's frozen pair input. All conditions for such a question preserve `y_suff_strict`, receive
`y_suff_semantic=NA`, status `unevaluable`, and reason
`claim_exceeds_frozen_nli_pair_budget`, and retain claim-length and model-budget metadata. They are
excluded from primary semantic-target training, calibration, threshold selection/evaluation, and
confirmation sampling, but remain available for transparent exclusion accounting and strict-label
diagnostics. They are not treated as insufficient.

Claim truncation was rejected because it could remove evidence-bearing content. New claim
segmentation was rejected because it would alter the semantic procedure after model/threshold
selection. The scoring configuration SHA-256
`98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf` therefore remains unchanged;
the exclusion rule is stored and canonically hashed separately in
`configs/phase03_semantic_label_governance.json`. The existing 13,744 valid pair scores remain
eligible for exact input-identity validation and reuse.

No final semantic labels, TEST semantic aggregate, confirmation human evaluation, Phase 4 model,
calibration artifact, or policy artifact existed at recovery time.

## Completion after governance approval

- The original 13,744-row primary checkpoint and manifest were preserved with
  `.pre-semantic-unevaluable-policy` suffixes. Two cached scores belonged to the newly excluded
  questions; a governance-filtered checkpoint was seeded with the other 13,742 exact rows. The
  remaining 2,257 evaluable pairs were scored, reaching 15,999/15,999 without repeating a cached
  evaluable score. All three 482-row development caches were reused.
- The final label-governance canonical SHA-256 is
  `233c61c45c7ad75b876e81deeb290d546a13f034f1d18c82787aea678ba0f533`; it references the unchanged
  scoring SHA-256 `98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf`.
- The label artifact contains 6,180 initially eligible TRAIN+VALIDATION conditions: 6,156 are
  semantic-evaluable and 24 conditions from two questions are semantic-unevaluable with reason
  `claim_exceeds_frozen_nli_pair_budget`. Their strict labels remain present and semantic labels
  are null.
- A new 100-row, 100-question blinded confirmation pack was generated. It has zero overlap with
  the original 150 development questions and contains zero semantic-unevaluable cases. Human
  fields remain blank and the answer key remains separate and unevaluated.
- The governance-resume execution was recorded separately at 1,940.5 seconds of tool wall time.
  Earlier interrupted/resumed sessions remain represented by their individual logs and checkpoint
  timestamps; no combined uninterrupted duration is asserted.
- The Phase 3.6 checker passed, the final Python compile check passed, and the repository test suite
  passed 55 tests. TEST semantic scores/aggregates remain uncalculated and Phase 4 remains unstarted.
