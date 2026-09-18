"""Helpers for constructing auditable DPO preference pairs from model errors."""

from __future__ import annotations

from typing import Any

from .contracts import extract_json, validate_decision
from .tool_use_data import compare_decisions


def classify_model_error(raw_output: str, target: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Classify one model output against a labelled tool-use target.

    ``None`` means the output is semantically correct and should not become a
    preference pair.  The returned category is intentionally coarse: it is used
    for pair balancing and audit, not as an additional training label.
    """

    try:
        prediction = validate_decision(extract_json(raw_output))
    except (ValueError, TypeError):
        return "schema_invalid", None

    comparison = compare_decisions(prediction, target)
    if target["decision"] != "call_tool" and prediction.decision == "call_tool":
        return "false_tool_call", comparison
    if not comparison["decision_correct"]:
        return "decision_mismatch", comparison
    if target["decision"] == "call_tool" and not comparison["tool_correct"]:
        return "tool_mismatch", comparison
    if target["decision"] == "call_tool" and not comparison["arguments_exact"]:
        return "argument_mismatch", comparison
    if not comparison["state_exact"]:
        return "state_mismatch", comparison
    return None, comparison
