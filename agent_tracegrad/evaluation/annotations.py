"""Externally supplied labels for true-failure evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel
from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class FailureAnnotation:
    annotation_id: str
    target_node_ids: Sequence[str]
    source: str = "human"
    trace_id: str | None = None
    trace_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.annotation_id:
            raise ValueError("annotation_id is required")
        node_ids = tuple(self.target_node_ids)
        if not node_ids:
            raise ValueError("target_node_ids must not be empty")
        if not self.source:
            raise ValueError("source is required")
        object.__setattr__(self, "target_node_ids", node_ids)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def to_label(self) -> GroundTruthLabel:
        return GroundTruthLabel(
            label_id=self.annotation_id,
            target_node_ids=self.target_node_ids,
            source=f"true-failure:{self.source}",
            metadata={
                **dict(self.metadata),
                "trace_id": self.trace_id,
                "trace_path": self.trace_path,
                "annotation_source": self.source,
            },
        )


def load_failure_annotations(path: str | Path) -> tuple[FailureAnnotation, ...]:
    """Load annotations from a JSON object/list file or JSONL records."""

    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    if source.suffix == ".jsonl":
        return tuple(_annotation_from_mapping(json.loads(line)) for line in text.splitlines() if line.strip())
    payload = json.loads(text)
    if isinstance(payload, list):
        return tuple(_annotation_from_mapping(item) for item in payload)
    if isinstance(payload, dict) and "annotations" in payload:
        return tuple(_annotation_from_mapping(item) for item in payload["annotations"])
    if isinstance(payload, dict):
        return (_annotation_from_mapping(payload),)
    raise ValueError("annotation file must contain an object, list, or JSONL records")


def labels_for_trace(
    annotations: Sequence[FailureAnnotation],
    trace: SerializedTrace,
    *,
    trace_path: str | None = None,
) -> tuple[GroundTruthLabel, ...]:
    labels = []
    trace_id = _trace_id(trace)
    for annotation in annotations:
        if annotation.trace_id is not None and annotation.trace_id != trace_id:
            continue
        if annotation.trace_path is not None and trace_path is not None and annotation.trace_path != trace_path:
            continue
        label = annotation.to_label()
        _validate_label_against_trace(label, trace)
        labels.append(label)
    return tuple(labels)


def _annotation_from_mapping(payload: Mapping[str, Any]) -> FailureAnnotation:
    if not isinstance(payload, Mapping):
        raise ValueError("annotation entries must be JSON objects")
    node_ids = payload.get("target_node_ids", payload.get("node_ids"))
    if isinstance(node_ids, str):
        node_ids = [node_ids]
    return FailureAnnotation(
        annotation_id=str(payload.get("annotation_id") or payload.get("label_id") or "annotation-1"),
        target_node_ids=tuple(str(node_id) for node_id in node_ids or ()),
        source=str(payload.get("source") or "human"),
        trace_id=str(payload["trace_id"]) if payload.get("trace_id") is not None else None,
        trace_path=str(payload["trace_path"]) if payload.get("trace_path") is not None else None,
        metadata=payload.get("metadata") or {},
    )


def _validate_label_against_trace(label: GroundTruthLabel, trace: SerializedTrace) -> None:
    missing = [node_id for node_id in label.target_node_ids if node_id not in trace.nodes]
    if missing:
        raise ValueError(f"annotation references missing node ids: {', '.join(missing)}")
    invalid = [
        node_id
        for node_id in label.target_node_ids
        if trace.nodes[node_id].block_role not in {"system", "user"}
    ]
    if invalid:
        raise ValueError("annotation target_node_ids must reference system or user nodes")


def _trace_id(trace: SerializedTrace) -> str | None:
    raw = trace.metadata.get("trace_id") or trace.metadata.get("source_trace_id")
    return str(raw) if raw is not None else None
