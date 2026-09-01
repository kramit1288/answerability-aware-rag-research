"""Frozen Phase 4 trajectory selection and Phase 5 generation policy views."""

from __future__ import annotations

from typing import Any


def select_policy_state(action: str) -> int | None:
    normalized = str(action).strip().upper()
    if normalized == "ANSWER_AT_K5":
        return 5
    if normalized == "ANSWER_AT_K10":
        return 10
    if normalized == "ABSTAIN":
        return None
    raise ValueError(f"unknown frozen Phase 4 trajectory action: {action!r}")


def build_policy_view(
    question_ids: list[str], states: dict[tuple[str, int], dict[str, Any]],
    *, policy_id: str, actions: dict[str, str] | None = None, fixed_k: int | None = None,
) -> list[dict[str, Any]]:
    if (actions is None) == (fixed_k is None):
        raise ValueError("provide exactly one of actions or fixed_k")
    rows: list[dict[str, Any]] = []
    for question_id in sorted(question_ids):
        action = "ANSWER_AT_K5" if fixed_k == 5 else "ANSWER_AT_K10" if fixed_k == 10 else actions[question_id]
        selected_k = select_policy_state(action)
        base = {
            "policy_id": policy_id,
            "question_id": question_id,
            "final_action": action,
            "answered": selected_k is not None,
            "selected_k": selected_k,
        }
        if selected_k is None:
            rows.append({**base, "response_id": None, "generation_status": None,
                         "rouge_l_f1": None, "bertscore_f1": None,
                         "unsupported_claim_rate": None, "fully_supported_response": None,
                         "response_with_any_unsupported_claim": None,
                         "mean_claim_support_score": None, "output_token_count": None})
        else:
            rows.append({**base, **states[(question_id, selected_k)]})
    return rows
