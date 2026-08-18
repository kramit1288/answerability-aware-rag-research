# Phase 3 context-sufficiency annotation guide

Version: `phase03-sufficiency-annotation-v1`

## Annotation unit

Judge one question and its complete ordered retrieved chunk bundle at the displayed retrieval
strategy and depth. Judge only the evidence shown in that bundle. Whole documents, search results,
automatic labels, model scores, and future policy decisions are outside the unit.

The annotation file may show the benchmark answer for benchmark-answerable questions. It shows no
answer for benchmark-impossible questions. Benchmark-impossible means that TechQA did not find an
answer in its candidate Technotes; it does not prove that no document anywhere could answer it.

## Labels

- `sufficient`: The retrieved text contains all material evidence needed to answer the question
  correctly and specifically. A reasonable technical answer can be recovered without adding an
  unsupported premise.
- `insufficient`: The bundle lacks required evidence, contains only the right topic/document, gives
  only a partial procedure or prerequisite, omits a material qualifier, version, number, negation,
  or constraint, or otherwise cannot support the benchmark answer.
- `ambiguous`: The display is malformed, contradictory, genuinely unclear, or requires specialist
  interpretation that the annotator cannot resolve from the supplied material. Do not use this
  merely because the wording differs from the benchmark answer.

## Evidence rules

Read every displayed chunk. Evidence may be split across more than one chunk and duplicate overlap
does not count against sufficiency. A correct filename or related passage is not sufficient by
itself. For procedures, required steps and prerequisites must be present. For negation, version,
numeric, configuration, and security questions, the decisive qualifier must be present. If chunks
conflict and the conflict cannot be resolved from the bundle, use `ambiguous` rather than guessing.

For benchmark-impossible questions, judge whether the retrieved bundle itself nevertheless appears
to contain enough evidence for a defensible answer. This is a contamination/anomaly audit, not a
request to search outside the shown bundle.

## Blinding and recording

Annotators must not view the automatic sufficiency label, evidence-coverage fraction, classifier
prediction, probability, or future answer/request/abstain action. Record:

- a genuine `annotator_id`;
- exactly one of `sufficient`, `insufficient`, or `ambiguous`;
- an optional concise rationale; and
- an ISO-8601 timestamp.

Annotators work independently before any adjudication. If two human files are available, raw
agreement and Cohen's kappa are calculated on their original categorical decisions. Codex, an LLM,
NLI, or an embedding model is never recorded as a human annotator.
