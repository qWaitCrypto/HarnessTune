from __future__ import annotations

import pytest

from agent_tracegrad.evaluation import PerturbationSpec, apply_trace_level_perturbation
from agent_tracegrad.trace import JsonTraceAdapter, TraceSerializer


class WhitespaceOffsetTokenizer:
    name_or_path = "whitespace-offset-tokenizer"

    def __call__(self, text: str, *, return_offsets_mapping: bool, add_special_tokens: bool) -> dict[str, object]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        offsets: list[tuple[int, int]] = []
        position = 0
        for part in text.split():
            start = text.index(part, position)
            end = start + len(part)
            offsets.append((start, end))
            position = end
        return {"offset_mapping": offsets}


def make_trace():
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "sys-1",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "Follow rules",
                "sequence_index": 0,
            },
            {
                "node_id": "user-1",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "alpha beta gamma delta",
                "sequence_index": 1,
            },
            {
                "node_id": "agent-1",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "wrong answer",
                "sequence_index": 2,
            },
        ]
    )
    serializer = TraceSerializer(WhitespaceOffsetTokenizer())
    return serializer.serialize(nodes, metadata={"trace_id": "trace-1"}), serializer


def make_swap_trace():
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "sys-1",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "Follow rules",
                "sequence_index": 0,
            },
            {
                "node_id": "user-1",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "first user content",
                "sequence_index": 1,
            },
            {
                "node_id": "user-2",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "second user content",
                "sequence_index": 2,
            },
            {
                "node_id": "agent-1",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "wrong answer",
                "sequence_index": 3,
            },
        ]
    )
    serializer = TraceSerializer(WhitespaceOffsetTokenizer())
    return serializer.serialize(nodes, metadata={"trace_id": "trace-2"}), serializer


def test_replace_with_placeholder_reserializes_trace_and_records_label() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(
        operator="replace_with_placeholder",
        target_node_ids=("user-1",),
        parameters={"placeholder": "masked content"},
    )

    result = apply_trace_level_perturbation(trace, spec, serializer)

    assert result.trace.nodes["user-1"].content == "masked content"
    assert result.trace.nodes["sys-1"].content == "Follow rules"
    assert result.label.target_node_ids == ("user-1",)
    assert result.label.source == "trace-level-perturbation"
    assert result.trace.metadata["trace_id"] == "trace-1"
    assert result.trace.metadata["perturbation"]["operator"] == "replace_with_placeholder"
    assert result.trace.nodes["user-1"].metadata["perturbation"]["original_content"] == "alpha beta gamma delta"


def test_truncate_uses_token_offsets_and_updates_spans() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(operator="truncate", target_node_ids=("user-1",), parameters={"ratio": 0.5})

    result = apply_trace_level_perturbation(trace, spec, serializer)

    assert result.trace.nodes["user-1"].content == "alpha beta"
    spans = {span.node_id: span for span in result.trace.spans}
    assert (spans["user-1"].start_token, spans["user-1"].end_token) == (2, 4)
    assert (spans["agent-1"].start_token, spans["agent-1"].end_token) == (4, 6)


def test_trace_level_perturbation_rejects_agent_target() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(operator="truncate", target_node_ids=("agent-1",), parameters={"ratio": 0.5})

    with pytest.raises(ValueError, match="must be in a system or user block"):
        apply_trace_level_perturbation(trace, spec, serializer)


def test_replace_with_placeholder_requires_configured_placeholder() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(operator="replace_with_placeholder", target_node_ids=("user-1",))

    with pytest.raises(ValueError, match="requires non-empty string"):
        apply_trace_level_perturbation(trace, spec, serializer)


def test_truncate_rejects_invalid_ratio() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(operator="truncate", target_node_ids=("user-1",), parameters={"ratio": 1.0})

    with pytest.raises(ValueError, match="ratio"):
        apply_trace_level_perturbation(trace, spec, serializer)


def test_trace_level_perturbation_rejects_unknown_operator() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(operator="missing", target_node_ids=("user-1",))

    with pytest.raises(ValueError, match="unknown perturbation operator"):
        apply_trace_level_perturbation(trace, spec, serializer)


def test_contradict_downstream_replaces_literal_once() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(
        operator="contradict_downstream",
        target_node_ids=("user-1",),
        parameters={"original": "beta", "replacement": "omega"},
    )

    result = apply_trace_level_perturbation(trace, spec, serializer)

    assert result.trace.nodes["user-1"].content == "alpha omega gamma delta"
    assert result.label.target_node_ids == ("user-1",)


def test_inject_unrelated_content_appends_configured_text() -> None:
    trace, serializer = make_trace()
    spec = PerturbationSpec(
        operator="inject_unrelated_content",
        target_node_ids=("sys-1",),
        parameters={"content": "unrelated zeta", "separator": " "},
    )

    result = apply_trace_level_perturbation(trace, spec, serializer)

    assert result.trace.nodes["sys-1"].content == "Follow rules unrelated zeta"
    assert result.trace.nodes["agent-1"].content == "wrong answer"


def test_swap_between_instances_swaps_two_configured_nodes() -> None:
    trace, serializer = make_swap_trace()
    spec = PerturbationSpec(
        operator="swap_between_instances",
        target_node_ids=("user-1", "user-2"),
        parameters={
            "replacements": {
                "user-1": "second user content",
                "user-2": "first user content",
            }
        },
    )

    result = apply_trace_level_perturbation(trace, spec, serializer)

    assert result.trace.nodes["user-1"].content == "second user content"
    assert result.trace.nodes["user-2"].content == "first user content"
    assert result.label.target_node_ids == ("user-1", "user-2")
