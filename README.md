# Answerability-Aware RAG Research

This repository is the working implementation for the MSc research:

**Answerability-Aware Evaluation of Hallucination and Reliability in Retrieval-Augmented Generation for Technical Documentation Question Answering**

The original Jupyter notebook is preserved under `notebooks/original_prototype.ipynb`. Do not keep extending that notebook as the final implementation. Use it as a reference while moving the experiment into reproducible Python modules and scripts.

## First-time setup

### 1. Initialize Git

```bash
git init
git add .
git commit -m "Initial research scaffold"
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For GPU-heavy generation/embedding experiments, Google Colab or Kaggle can still be used. The local repository remains the source of truth.

### 3. Start Codex from the repository root

```bash
codex
```

Before asking Codex to implement anything, tell it to read:

- `AGENTS.md`
- `docs/RESEARCH_IMPLEMENTATION_PLAN.md`
- `docs/MARKER_FEEDBACK.md`
- `notebooks/original_prototype.ipynb`

Then use the Phase 0 prompt in `docs/CODEX_PROMPTS.md`.

## Recommended workflow

Do **not** ask Codex to "finish the whole thesis implementation" in one task.

Use these milestones:

1. Phase 0: audit and freeze the experimental contract.
2. Phase 1: data loading, official corpus, group-aware splits, dataset manifests.
3. Phase 2: BM25, dense and hybrid retrieval experiments.
4. Phase 3: retrieved-context sufficiency labels and validation.
5. Phase 4: baselines, calibrated classifier and two-threshold policy.
6. Phase 5: generation and unsupported-claim evaluation.
7. Phase 6: statistics, confidence intervals, tables and figures.
8. Phase 7: final reproducibility run and thesis-ready result bundle.

Commit after every phase.

## Core rule

**Never fabricate or silently fill experimental results.** If an experiment has not run successfully, the result is unknown.

## Output locations

- `artifacts/data/` — immutable split manifests and query-level datasets
- `artifacts/results/` — CSV/JSON result tables
- `artifacts/figures/` — thesis-ready figures
- `artifacts/models/` — trained/calibrated model artifacts
- `reports/` — generated summaries if added later

