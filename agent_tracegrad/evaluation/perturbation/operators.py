"""Domain-abstract perturbation operators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from agent_tracegrad.trace.schema import TraceNode
from agent_tracegrad.trace.serializer import OffsetTokenizer

PerturbationOperator = Callable[[TraceNode, Mapping[str, Any], OffsetTokenizer], str]


def replace_with_placeholder(node: TraceNode, parameters: Mapping[str, Any], tokenizer: OffsetTokenizer) -> str:
    del node, tokenizer
    placeholder = parameters.get("placeholder")
    if not isinstance(placeholder, str) or not placeholder:
        raise ValueError("replace_with_placeholder requires non-empty string parameter 'placeholder'")
    return placeholder


def truncate(node: TraceNode, parameters: Mapping[str, Any], tokenizer: OffsetTokenizer) -> str:
    ratio = parameters.get("ratio")
    if not isinstance(ratio, int | float) or not 0.0 < float(ratio) < 1.0:
        raise ValueError("truncate requires numeric parameter 'ratio' in (0, 1)")
    offsets = tuple(tokenizer(node.content, return_offsets_mapping=True, add_special_tokens=False).get("offset_mapping") or ())
    if not offsets:
        return ""
    keep_tokens = max(1, int(len(offsets) * float(ratio)))
    keep_tokens = min(keep_tokens, len(offsets))
    end_char = _coerce_offset(offsets[keep_tokens - 1])[1]
    return node.content[:end_char]


OPERATORS: Mapping[str, PerturbationOperator] = {
    "replace_with_placeholder": replace_with_placeholder,
    "truncate": truncate,
}


def get_operator(name: str) -> PerturbationOperator:
    try:
        return OPERATORS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(OPERATORS))
        raise ValueError(f"unknown perturbation operator {name!r}; expected one of: {allowed}") from exc


def _coerce_offset(offset: Any) -> tuple[int, int]:
    if not isinstance(offset, tuple | list) or len(offset) != 2:
        raise ValueError("tokenizer offsets must be two-item sequences")
    start, end = int(offset[0]), int(offset[1])
    if start < 0 or end < start:
        raise ValueError("tokenizer offsets must be non-negative half-open ranges")
    return start, end
