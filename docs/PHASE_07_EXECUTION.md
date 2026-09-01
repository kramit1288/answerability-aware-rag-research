# Phase 7 Execution Protocol

## Scope

Phase 7 performs the final one-time TechQA TEST evaluation of the frozen Phase 3 target, Phase 4
model and policies, Phase 5 generator and grounding evaluator, and Phase 6 statistical framework.
No fitting, tuning, threshold selection, prompt change, evaluator change, operating-point
selection, thesis rewrite, commit, or tag belongs to this phase.

## Pre-unseal stage gates

1. Read the experiment contract and complete Phase 3-6 governance, execution, results, configs,
   manifests, and relevant implementation paths without reading TechQA TEST scientific content.
2. Create the Phase 7 configuration, research decisions, execution protocol, and artifact schemas.
3. Canonically hash the configuration and physically hash all four governance files into
   `artifacts/results/phase07_pre_test_governance_freeze.json`. It must state that no TEST
   scientific content was accessed and contain the exact sentence: "No post-TEST scientific
   choice is permitted."
4. Independently reproduce the Phase 3 target, Phase 4 model/policy/binary, Phase 5 manifest,
   generator/prompt/config/evaluator/threshold, and Phase 6 config/manifest hashes. Verify the
   Phase 6 tag and current branch ancestry. Any mismatch stops before TEST access.
5. Write `artifacts/results/phase07_test_unseal_record.json` with an exact UTC-offset timestamp,
   current commit/branch, upstream tag, Phase 7 configuration SHA, verified hashes, and required
   no-prior-selection/no-post-unseal-tuning declarations. Writing this record is the unseal event.

## Post-unseal execution

1. Reproduce and persist the frozen TEST census and primary/sensitivity exclusion manifest.
2. Materialize strict TEST labels, score the frozen semantic method, apply the frozen input-budget
   exclusion and strict-preserving rescue rule, then close and hash the immutable TEST target
   before inspecting class balance.
3. Construct exactly 39 TEST features. Run registry/leakage checks, load the frozen serialized
   TRAIN-fitted pipeline, and infer probabilities without any fit call.
4. Produce classifier metrics, reliability bins, 5,000 question-bootstrap intervals, the complete
   risk-coverage curve/AURC, and observations at the three frozen thresholds.
5. Simulate the frozen primary and sensitivity adaptive policies on hybrid k5/k10, retaining exact
   trajectories, metrics, and question-bootstrap intervals.
6. Assemble and hash every hybrid k5/k10 prompt-visible context. Generate each state once with the
   frozen model and cache key. Close and hash generation before loading references.
7. Compute frozen ROUGE-L/BERTScore, claims, grounding scores, response grounding, G0-G3 views,
   and paired state records. Failures remain explicit; abstention quality stays NA.
8. Run the frozen Phase 6 families and Holm correction, construct the benchmark-impossible
   sensitivity, and create the descriptive VALIDATION-to-TEST comparison.
9. Generate final RQ evidence, tables, and only the materially useful TEST-aware figures.
10. Run the post-TEST tuning guard, add Phase-7-scoped LF/binary rules, and create the independent
    manifest without self-reference.

## Failure and resumption

After the unseal record exists, TEST is never described as resealed. Deterministic checkpoints may
resume incomplete computation only when their complete scientific cache identity matches. A valid
completed state is never regenerated. One infrastructure retry is allowed only where Phase 5
already allowed it, with byte-identical inputs/settings and a recorded attempt. Scientific output
quality never triggers a retry. Any departure from the frozen procedure stops and is reported; it
does not authorize a substitute.

## Intended commands

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe scripts/freeze_phase07_governance.py --config configs/phase07_final_test.json
.venv/Scripts/python.exe scripts/check_phase07.py --pre-unseal
.venv/Scripts/python.exe scripts/unseal_phase07_test.py --config configs/phase07_final_test.json
.venv/Scripts/python.exe scripts/run_phase07_final_test.py --config configs/phase07_final_test.json
.venv/Scripts/python.exe scripts/check_phase07.py
.venv/Scripts/python.exe -m pytest --basetemp=.pytest-local-temp
.venv/Scripts/python.exe -m compileall src scripts tests
git -c safe.directory=C:/answerability-aware-rag-research diff --check
```

The pytest temporary directory is removed afterward. Historical boundary checkers are not changed
merely because they correctly reject Phase 7 existence.

## Completion boundary

Completion requires every acceptance criterion in the frozen configuration and user instruction,
including `post_test_tuning_detected=false`. Stop after the final TEST report and wait for human
review. Do not commit or tag.
