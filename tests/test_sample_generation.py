from __future__ import annotations

import pytest

from agent_tracegrad.evaluation import generate_trace_level_samples
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
                "content": "first content",
                "sequence_index": 1,
            },
            {
                "node_id": "user-2",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "second content",
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
    return serializer.serialize(nodes, metadata={"trace_id": "trace-1"}), serializer


def test_generate_trace_level_samples_iterates_system_and_user_only() -> None:
    trace, serializer = make_trace()

    samples = generate_trace_level_samples(
        trace,
        serializer,
        operator_configs=(
            {
                "operator": "replace_with_placeholder",
                "parameters": {"placeholder": "masked"},
            },
        ),
    )

    assert [sample.spec.target_node_ids for sample in samples] == [("sys-1",), ("user-1",), ("user-2",)]
    assert all("agent-1" not in sample.spec.target_node_ids for sample in samples)
    assert samples[0].perturbation.trace.nodes["sys-1"].content == "masked"


def test_generate_trace_level_samples_respects_max_samples() -> None:
    trace, serializer = make_trace()

    samples = generate_trace_level_samples(
        trace,
        serializer,
        operator_configs=(
            {
                "operator": "inject_unrelated_content",
                "parameters": {"content": "noise"},
            },
        ),
        max_samples=2,
    )

    assert len(samples) == 2


def test_generate_trace_level_samples_builds_swap_sample_for_same_kind_pair() -> None:
    trace, serializer = make_trace()

    samples = generate_trace_level_samples(
        trace,
        serializer,
        operator_configs=({"operator": "swap_between_instances"},),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.spec.target_node_ids == ("user-1", "user-2")
    assert sample.perturbation.trace.nodes["user-1"].content == "second content"
    assert sample.perturbation.trace.nodes["user-2"].content == "first content"
    assert sample.perturbation.label.target_node_ids == ("user-1", "user-2")


def test_generate_trace_level_samples_rejects_invalid_max_samples() -> None:
    trace, serializer = make_trace()

    with pytest.raises(ValueError, match="max_samples"):
        generate_trace_level_samples(trace, serializer, operator_configs=(), max_samples=0)
