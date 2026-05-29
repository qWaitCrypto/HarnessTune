from __future__ import annotations

import pytest

from agent_tracegrad.attribution import AttributionResult
from agent_tracegrad.evaluation import GroundTruthLabel, PerturbationSpec
from agent_tracegrad.target import FailureTarget
from agent_tracegrad.trace import JsonTraceAdapter, SpanMetadata, SerializedTrace, TraceNode, TraceSerializer


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


def make_trace() -> SerializedTrace:
    nodes = {
        "sys-1": TraceNode(
            node_id="sys-1",
            block_role="system",
            sub_block_kind="system.instruction",
            content="Follow the rules.",
        ),
        "user-1": TraceNode(
            node_id="user-1",
            block_role="user",
            sub_block_kind="user.content",
            content="Do the task.",
        ),
        "agent-1": TraceNode(
            node_id="agent-1",
            block_role="agent",
            sub_block_kind="agent.tool_call",
            content='{"name":"bad_call"}',
        ),
    }
    return SerializedTrace(
        nodes=nodes,
        serialized_text="Follow the rules. Do the task. bad_call",
        spans=(
            SpanMetadata("span-sys-1", "sys-1", "system", "system.instruction", 0, 3),
            SpanMetadata("span-user-1", "user-1", "user", "user.content", 3, 6),
            SpanMetadata("span-agent-1", "agent-1", "agent", "agent.tool_call", 6, 8),
        ),
        tokenizer_name="test-tokenizer",
    )


def test_trace_node_accepts_valid_role_kind_pair() -> None:
    node = TraceNode(
        node_id="n1",
        block_role="system",
        sub_block_kind="system.skills",
        content="Skill text",
        metadata={"turn": 1},
        parents=["root"],
    )

    assert node.node_id == "n1"
    assert node.metadata["turn"] == 1
    assert node.parents == ("root",)


def test_trace_node_rejects_invalid_sub_block_kind() -> None:
    with pytest.raises(ValueError, match="invalid sub_block_kind"):
        TraceNode(
            node_id="n1",
            block_role="system",
            sub_block_kind="system.memory",
            content="Bad kind",
        )


def test_trace_node_rejects_role_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="not valid for block_role"):
        TraceNode(
            node_id="n1",
            block_role="system",
            sub_block_kind="user.content",
            content="Bad pair",
        )


def test_serialized_trace_rejects_span_role_kind_mismatch() -> None:
    node = TraceNode("agent-1", "agent", "agent.content", "Done")
    with pytest.raises(ValueError, match="does not match node"):
        SerializedTrace(
            nodes={"agent-1": node},
            serialized_text="Done",
            spans=(SpanMetadata("span-1", "agent-1", "user", "user.content", 0, 1),),
            tokenizer_name="test-tokenizer",
        )


def test_failure_target_accepts_agent_node() -> None:
    trace = make_trace()
    target = FailureTarget(target_id="target-1", node_ids=("agent-1",), span=(6, 8))

    target.validate_against_trace(trace)


def test_failure_target_rejects_system_or_user_node() -> None:
    trace = make_trace()
    target = FailureTarget(target_id="target-1", node_ids=("user-1",))

    with pytest.raises(ValueError, match="must be in an agent block"):
        target.validate_against_trace(trace)


def test_failure_target_rejects_span_outside_agent_node() -> None:
    trace = make_trace()
    target = FailureTarget(target_id="target-1", node_ids=("agent-1",), span=(0, 1))

    with pytest.raises(ValueError, match="span must fall inside"):
        target.validate_against_trace(trace)


def test_perturbation_spec_accepts_system_or_user_node() -> None:
    trace = make_trace()
    spec = PerturbationSpec(operator="truncate", target_node_ids=("user-1",), parameters={"ratio": 0.5})

    spec.validate_against_trace(trace)


def test_perturbation_spec_rejects_agent_node() -> None:
    trace = make_trace()
    spec = PerturbationSpec(operator="truncate", target_node_ids=("agent-1",), parameters={"ratio": 0.5})

    with pytest.raises(ValueError, match="must be in a system or user block"):
        spec.validate_against_trace(trace)


def test_attribution_result_validates_length_and_zero_agent_scores() -> None:
    trace = make_trace()
    result = AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0),
    )

    result.validate_against_trace(trace)


def test_attribution_result_rejects_nonzero_agent_scores() -> None:
    trace = make_trace()
    result = AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.0),
    )

    with pytest.raises(ValueError, match="agent-range token scores must be zero"):
        result.validate_against_trace(trace)


def test_attribution_result_rejects_wrong_same_model_flag() -> None:
    with pytest.raises(ValueError, match="same_model must match"):
        AttributionResult(
            method_name="gradient",
            attribution_model_name="model-a",
            execution_model_name="model-b",
            same_model=True,
            target_id="target-1",
            token_scores=(0.0,),
        )


def test_ground_truth_label_requires_targets() -> None:
    with pytest.raises(ValueError, match="target_node_ids must not be empty"):
        GroundTruthLabel(label_id="label-1", target_node_ids=(), source="trace-level-perturbation")


def test_json_trace_adapter_accepts_normalized_mapping() -> None:
    adapter = JsonTraceAdapter()

    nodes = adapter.adapt(
        {
            "nodes": [
                {
                    "node_id": "agent-1",
                    "block_role": "agent",
                    "sub_block_kind": "agent.content",
                    "content": "Done",
                    "sequence_index": 2,
                },
                {
                    "node_id": "sys-1",
                    "block_role": "system",
                    "sub_block_kind": "system.instruction",
                    "content": "Follow rules",
                    "sequence_index": 1,
                },
            ]
        }
    )

    assert [node.node_id for node in nodes] == ["sys-1", "agent-1"]


def test_json_trace_adapter_rejects_missing_required_field() -> None:
    adapter = JsonTraceAdapter()

    with pytest.raises(ValueError, match="missing required field"):
        adapter.adapt({"nodes": [{"node_id": "n1", "block_role": "system", "content": "Missing kind"}]})


def test_trace_serializer_builds_token_aligned_spans() -> None:
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "sys-1",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "Follow rules",
            },
            {
                "node_id": "agent-1",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "Done now",
            },
        ]
    )
    serializer = TraceSerializer(WhitespaceOffsetTokenizer())

    trace = serializer.serialize(nodes, metadata={"trace_id": "fixture-1"})

    assert trace.serialized_text == "Follow rules\nDone now"
    assert trace.tokenizer_name == "whitespace-offset-tokenizer"
    assert trace.metadata["trace_id"] == "fixture-1"
    assert [(span.node_id, span.start_token, span.end_token) for span in trace.spans] == [
        ("sys-1", 0, 2),
        ("agent-1", 2, 4),
    ]
