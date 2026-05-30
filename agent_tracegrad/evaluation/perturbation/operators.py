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


def contradict_downstream(node: TraceNode, parameters: Mapping[str, Any], tokenizer: OffsetTokenizer) -> str:
    del tokenizer
    original = parameters.get("original")
    replacement = parameters.get("replacement")
    if not isinstance(original, str) or not original:
        raise ValueError("contradict_downstream requires non-empty string parameter 'original'")
    if not isinstance(replacement, str) or not replacement:
        raise ValueError("contradict_downstream requires non-empty string parameter 'replacement'")
    if original == replacement:
        raise ValueError("contradict_downstream requires distinct 'original' and 'replacement' parameters")
    if original not in node.content:
        raise ValueError("contradict_downstream original text was not found in target node content")
    return node.content.replace(original, replacement, 1)


def inject_unrelated_content(node: TraceNode, parameters: Mapping[str, Any], tokenizer: OffsetTokenizer) -> str:
    del tokenizer
    content = parameters.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("inject_unrelated_content requires non-empty string parameter 'content'")
    separator = parameters.get("separator", "\n")
    if not isinstance(separator, str):
        raise ValueError("inject_unrelated_content parameter 'separator' must be a string")
    return f"{node.content}{separator}{content}"


def swap_between_instances(node: TraceNode, parameters: Mapping[str, Any], tokenizer: OffsetTokenizer) -> str:
    del tokenizer
    replacements = parameters.get("replacements")
    if not isinstance(replacements, Mapping):
        raise ValueError("swap_between_instances requires mapping parameter 'replacements'")
    replacement = replacements.get(node.node_id)
    if not isinstance(replacement, str):
        raise ValueError(f"swap_between_instances missing string replacement for node {node.node_id!r}")
    return replacement


OPERATORS: Mapping[str, PerturbationOperator] = {
    "contradict_downstream": contradict_downstream,
    "inject_unrelated_content": inject_unrelated_content,
    "replace_with_placeholder": replace_with_placeholder,
    "swap_between_instances": swap_between_instances,
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
