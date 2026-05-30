"""Thin orchestration API for evaluation sample generation from raw traces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel
from agent_tracegrad.evaluation.sample_generation import TraceLevelSample, generate_trace_level_samples, samples_from_labels
from agent_tracegrad.target.marker import FailureTargetMarker
from agent_tracegrad.target.objective import TargetObjective
from agent_tracegrad.target.registry import get_failure_target_marker
from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.adapter import TraceAdapter
from agent_tracegrad.trace.ingest import ingest_trace
from agent_tracegrad.trace.schema import SerializedTrace
from agent_tracegrad.trace.serializer import TraceSerializer


@dataclass(frozen=True)
class TraceEvaluationContext:
    trace: SerializedTrace
    targets: Sequence[FailureTarget]
    objective: TargetObjective
    samples: Sequence[TraceLevelSample]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def generate_evaluation_context(
    raw_trace: Any,
    *,
    tokenizer: Any,
    input_format: str = "json-fixture",
    target_node_ids: Sequence[str] | None = None,
    target_marker: str | FailureTargetMarker | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    objective: TargetObjective | None = None,
    operator_configs: Sequence[Mapping[str, Any]],
    annotation_labels: Sequence[GroundTruthLabel] = (),
    max_samples: int | None = None,
    trace_metadata: Mapping[str, Any] | None = None,
    adapter: TraceAdapter | None = None,
) -> TraceEvaluationContext:
    """Compose ingestion, target resolution, and canonical sample generation."""

    trace = ingest_trace(
        raw_trace,
        input_format=input_format,
        tokenizer=tokenizer,
        trace_metadata=trace_metadata,
        adapter=adapter,
    )
    serializer = TraceSerializer(tokenizer)
    target = _resolve_target(
        trace,
        target_node_ids=target_node_ids,
        target_marker=target_marker,
        target_id=target_id,
        target_span=target_span,
    )
    target.validate_against_trace(trace)
    resolved_objective = _resolve_objective(objective, target)
    resolved_objective.validate_against_trace(trace)
    samples = samples_from_labels(trace, annotation_labels)
    if operator_configs:
        samples = samples + generate_trace_level_samples(
            trace,
            serializer,
            operator_configs=operator_configs,
            max_samples=max_samples,
        )
    if not samples:
        raise ValueError("evaluation requires at least one operator config or annotation label")
    return TraceEvaluationContext(
        trace=trace,
        targets=(target,),
        objective=resolved_objective,
        samples=samples,
        metadata={
            "input_format": input_format,
            "objective_type": resolved_objective.objective_type,
            "target_count": 1,
            "sample_count": len(samples),
        },
    )


def _resolve_target(
    trace: SerializedTrace,
    *,
    target_node_ids: Sequence[str] | None,
    target_marker: str | FailureTargetMarker | None,
    target_id: str,
    target_span: tuple[int, int] | None,
) -> FailureTarget:
    if target_node_ids:
        return FailureTarget(target_id=target_id, node_ids=target_node_ids, span=target_span)
    marker = _coerce_marker(target_marker or "last-agent-output")
    targets = tuple(marker.mark(trace))
    if not targets:
        raise ValueError(f"failure target marker {marker.name!r} did not produce any targets")
    if len(targets) > 1:
        raise ValueError("evaluation sample generation currently expects exactly one failure target")
    target = targets[0]
    return FailureTarget(
        target_id=target_id,
        node_ids=target.node_ids,
        span=target_span if target_span is not None else target.span,
    )


def _coerce_marker(marker: str | FailureTargetMarker) -> FailureTargetMarker:
    if isinstance(marker, str):
        return get_failure_target_marker(marker)
    return marker


def _resolve_objective(objective: TargetObjective | None, target: FailureTarget) -> TargetObjective:
    if objective is None:
        return TargetObjective.bad_action(target)
    if objective.bad_target is not None:
        return objective
    if objective.objective_type == "expected_action" and objective.expected_target is not None:
        return TargetObjective(
            objective_id=objective.objective_id,
            objective_type="expected_action",
            bad_target=target,
            expected_target=objective.expected_target,
            source=objective.source,
            metadata=dict(objective.metadata),
        )
    if objective.objective_type == "contrastive" and objective.expected_target is not None:
        return TargetObjective.contrastive(
            target,
            objective.expected_target,
            objective_id=objective.objective_id,
            source=objective.source,
            metadata={
                key: value
                for key, value in dict(objective.metadata).items()
                if key != "requires_resolved_bad_target"
            },
        )
    return objective
