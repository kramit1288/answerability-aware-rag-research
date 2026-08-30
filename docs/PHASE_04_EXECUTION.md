# Phase 4 Execution Protocol

## Scope

Phase 4 constructs leakage-safe inference-time features, trains lightweight predictors of the frozen retrieval-conditioned context-sufficiency target, evaluates discrimination and calibration, constructs risk–coverage results, and freezes a hybrid k=5 to optional k=10 three-way policy. It does not generate answers and does not begin Phase 5.

The authoritative decisions are `docs/PHASE_04_RESEARCH_DECISIONS.md` and `configs/phase04_modeling.json`. The frozen `docs/RESEARCH_DECISIONS.md` is an immutable Phase 3 artifact and is not extended.

## Stage gates

1. Run all Phase 1–3 integrity checkers and independently verify the frozen upstream hashes and PRIMARY population.
2. Freeze and hash the Phase 4 registry, modeling configuration, execution protocol, artifact schemas, and Phase 4 decision log before calculating validation performance.
3. Generate features for TRAIN and VALIDATION only. Fail before feature generation or inference for TEST.
4. Run grouped TRAIN development for B1, Logistic Regression, Random Forest, and grouped OOF Random Forest probabilities.
5. Refit selected candidates on all TRAIN rows and calculate VALIDATION metrics. Fit no transformation or calibrator on VALIDATION.
6. Select and hash the model configuration before any policy threshold tuning.
7. Produce feature analysis, frozen ablations, full risk–coverage data, AURC, and 5/10/20% operating points.
8. Enumerate the complete hybrid k=5 to optional k=10 policy grid and compare P0, P1, and P2. Freeze the 10% illustrative policy.
9. Only then run the separately reported benchmark-impossible sensitivity analysis with the PRIMARY architecture unchanged.
10. Build the independent Phase 4 artifact manifest and integrity report; run the full validation suite; stop for human review.

If implementing the frozen specification is impossible because of a genuine defect, document the defect and stop before changing the search space or inspecting further VALIDATION results. If any change would mutate a Phase 1–3 frozen artifact, stop and request authorization.

## Canonical commands

The intended commands are:

```powershell
python scripts/run_phase04_modeling.py --config configs/phase04_modeling.json
python scripts/check_phase04.py
```

Upstream checker commands are discovered from their existing script interfaces and run unchanged. The Phase 3.6b checker uses `PYTHONPATH=src` when required by its documented invocation.

## Reproducibility and fitting boundary

Randomness is fixed by seed 42. Every grouped fold persists question and condition counts, class counts, and a zero-question-overlap assertion. Feature-score normalization, imputation, one-hot handling, and scaling live inside the fitted pipeline. Calibration mappings are fit to grouped OOF TRAIN predictions. Bootstrap resamples question IDs rather than condition rows.

All JSON outputs use deterministic sorted-key serialization with a trailing LF. CSV outputs use UTF-8 and LF. Parquet feature artifacts use the repository's frozen deterministic PyArrow settings. Model binary hashes are recorded with the producing library versions and recreation command.

## Interpretation boundary

The predictor estimates the Phase 3 operational target for retrieval-conditioned context sufficiency. Its probability is not epistemic certainty. Similarity features are not hallucination detection. Feature importance is not causal. The 10% risk constraint is an illustrative VALIDATION operating point, not a universal or production guarantee.

## Completion boundary

Completion requires every artifact and test listed in the approved specification, an independently verifiable Phase 4 manifest, a passing Phase 4 checker, preserved upstream hashes, and a sealed TEST partition. No commit or tag is created automatically.
