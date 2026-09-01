# Phase 6 Execution Protocol

## Scope

Phase 6 verifies frozen upstream state, performs validation-only statistical inference, produces
confidence intervals and numerical effect sizes, and generates thesis-ready tables and figures.
It does not access TechQA TEST, rerun generation or NLI, retrain the Phase 4 model, modify any
threshold, or begin Phase 7.

## Stage gates

1. Verify the Phase 3 target, Phase 4 selected model/policy, and Phase 5 manifest hashes.
2. Reproduce 89 TechQA VALIDATION questions, 178 generated states, 42,296 RAGTruth claims, and
   `t_support=0.16`; verify the Phase 5 manifest and TechQA TEST seal.
3. Freeze the Phase 6 configuration, research decisions, execution protocol, schemas, exact RQ
   matrix, test families, effect sizes, bootstrap, missingness, correction, tables, figures, and
   interpretation rules before running inferential analysis.
4. Run the Phase 6 pipeline once from frozen Phase 1-5 artifacts. First reproduce the five frozen
   k10-minus-k5 descriptive mean differences; stop before inference if any differs.
5. Calculate Families A-C, apply Holm within family, and persist tests, effects, intervals,
   transition tables, group summaries, and policy intervals.
6. Consolidate Phase 2-5 results into the eleven predeclared tables and generate every figure from
   a persisted figure-data CSV.
7. Create the post-results summary only after all statistical artifacts exist.
8. Add Phase-6-scoped LF/binary rules before final manifest generation. Build the Phase 6
   manifest, integrity report, and checker output.
9. Run the Phase 6 checker, full pytest suite with local temp root, compileall, and Git diff check.
   Remove the local pytest temp directory and stop for review without commit or tag.

## Intended commands

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe scripts/run_phase06_statistics.py --config configs/phase06_statistics.json
.venv/Scripts/python.exe scripts/check_phase06.py
.venv/Scripts/python.exe -m pytest --basetemp=.pytest-local-temp
.venv/Scripts/python.exe -m compileall src scripts tests
git -c safe.directory=C:/answerability-aware-rag-research diff --check
```

## Reproducibility

The master seed is 42. TechQA bootstrap sampling is by `question_id`; RAGTruth intervals retain
the frozen `source_id` grouping. JSON is sorted UTF-8 with trailing LF, CSV uses stable columns
and LF, SVG uses a fixed Matplotlib hash salt and no generation timestamp, and PNG is saved at
300 DPI. Source artifacts are read-only.

## Failure and interpretation rules

Any upstream hash mismatch, descriptive-difference mismatch, paired-alignment failure, missing
Holm family member, independent-row bootstrap, NA-to-zero conversion, TechQA TEST access, or
Phase 7 artifact causes failure. A non-significant result is not evidence of equality. Automatic
grounding remains an imperfect proxy, and all policy quality is interpreted with coverage.
