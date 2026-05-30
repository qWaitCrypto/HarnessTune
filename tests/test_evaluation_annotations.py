from __future__ import annotations

import json

import pytest

from agent_tracegrad.evaluation import FailureAnnotation, labels_for_trace, load_failure_annotations
from agent_tracegrad.trace import JsonTraceAdapter, TraceSerializer

from tests.test_single_trace_analysis import WhitespaceOffsetTokenizer


def _trace():
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "policy",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "policy text",
                "sequence_index": 0,
            },
            {
                "node_id": "user",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "user clue",
                "sequence_index": 1,
            },
            {
                "node_id": "agent",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "wrong answer",
                "sequence_index": 2,
            },
        ]
    )
    return TraceSerializer(WhitespaceOffsetTokenizer()).serialize(nodes, metadata={"trace_id": "trace-a"})


def test_load_failure_annotations_from_json_list(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "annotation_id": "ann-1",
                    "trace_id": "trace-a",
                    "target_node_ids": ["policy"],
                    "source": "human",
                }
            ]
        ),
        encoding="utf-8",
    )

    annotations = load_failure_annotations(path)

    assert annotations == (
        FailureAnnotation(
            annotation_id="ann-1",
            trace_id="trace-a",
            target_node_ids=("policy",),
            source="human",
        ),
    )


def test_labels_for_trace_filters_and_validates_annotations() -> None:
    labels = labels_for_trace(
        (
            FailureAnnotation(annotation_id="ann-1", trace_id="trace-a", target_node_ids=("policy",)),
            FailureAnnotation(annotation_id="ann-2", trace_id="other", target_node_ids=("user",)),
        ),
        _trace(),
    )

    assert len(labels) == 1
    assert labels[0].label_id == "ann-1"
    assert labels[0].source == "true-failure:human"


def test_labels_for_trace_rejects_agent_annotation_target() -> None:
    with pytest.raises(ValueError, match="system or user"):
        labels_for_trace(
            (FailureAnnotation(annotation_id="ann-1", target_node_ids=("agent",)),),
            _trace(),
        )
