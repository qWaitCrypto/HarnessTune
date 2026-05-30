"""Trace-level perturbation utilities."""

from agent_tracegrad.evaluation.perturbation.operators import (
    PerturbationOperator,
    contradict_downstream,
    get_operator,
    inject_unrelated_content,
    insert_text,
    mask_jsonpath,
    replace_with_placeholder,
    remove_text_span,
    swap_between_instances,
    truncate,
)
from agent_tracegrad.evaluation.perturbation.trace_level import TraceLevelPerturbation, apply_trace_level_perturbation

__all__ = [
    "PerturbationOperator",
    "TraceLevelPerturbation",
    "apply_trace_level_perturbation",
    "contradict_downstream",
    "get_operator",
    "inject_unrelated_content",
    "insert_text",
    "mask_jsonpath",
    "replace_with_placeholder",
    "remove_text_span",
    "swap_between_instances",
    "truncate",
]
