"""Single-trace analysis pipeline for real-gradient attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.analysis.aggregation import AttributionDistribution, aggregate_attribution
from agent_tracegrad.analysis.ranking import RankedAttribution, rank_distribution
from agent_tracegrad.attribution.gradient import (
    GradientSaliencyAttribution,
    GradientTimesInputAttribution,
    IntegratedGradientsAttribution,
)
from agent_tracegrad.attribution.method import AttributionMethod
from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.marker import FailureTargetMarker
from agent_tracegrad.target.objective import TargetObjective, target_objective_to_dict
from agent_tracegrad.target.registry import get_failure_target_marker
from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.adapter import TraceAdapter
from agent_tracegrad.trace.registry import get_trace_adapter
from agent_tracegrad.trace.schema import SerializedTrace
from agent_tracegrad.trace.serializer import TraceSerializer


@dataclass(frozen=True)
class AnalysisRanking:
    grain: str
    view_name: str
    items: Sequence[RankedAttribution]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


@dataclass(frozen=True)
class SingleTraceAnalysisResult:
    trace: SerializedTrace
    target: FailureTarget
    attribution: AttributionResult
    distributions: Sequence[AttributionDistribution]
    rankings: Sequence[AnalysisRanking]
    metadata: Mapping[str, Any]
    objective: TargetObjective | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "distributions", tuple(self.distributions))
        object.__setattr__(self, "rankings", tuple(self.rankings))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def analyze_normalized_trace(
    raw_trace: Any,
    *,
    target_node_ids: Sequence[str],
    model: ModelAdapter,
    tokenizer: Any | None = None,
    method: str = "gradient_saliency",
    execution_model_name: str | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    topk_mean_k: int = 5,
    ranking_grain: str = "node",
    ranking_view: str = "sum",
    integrated_gradients_steps: int = 16,
    trace_metadata: Mapping[str, Any] | None = None,
) -> SingleTraceAnalysisResult:
    """Run the single-sample TraceGrad path over a normalized trace payload."""

    return analyze_trace(
        raw_trace,
        input_format="json-fixture",
        target_node_ids=target_node_ids,
        model=model,
        tokenizer=tokenizer,
        method=method,
        execution_model_name=execution_model_name,
        target_id=target_id,
        target_span=target_span,
        topk_mean_k=topk_mean_k,
        ranking_grain=ranking_grain,
        ranking_view=ranking_view,
        integrated_gradients_steps=integrated_gradients_steps,
        trace_metadata=trace_metadata,
    )


def analyze_trace(
    raw_trace: Any,
    *,
    input_format: str = "json-fixture",
    target_node_ids: Sequence[str] | None = None,
    target_marker: str | FailureTargetMarker | None = None,
    objective: TargetObjective | None = None,
    model: ModelAdapter,
    tokenizer: Any | None = None,
    method: str = "gradient_saliency",
    execution_model_name: str | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    topk_mean_k: int = 5,
    ranking_grain: str = "node",
    ranking_view: str = "sum",
    integrated_gradients_steps: int = 16,
    trace_metadata: Mapping[str, Any] | None = None,
    adapter: TraceAdapter | None = None,
) -> SingleTraceAnalysisResult:
    """Run the single-sample TraceGrad path over any registered trace format."""

    trace_adapter = adapter or get_trace_adapter(input_format)
    nodes = trace_adapter.adapt(raw_trace)
    serializer = TraceSerializer(tokenizer or model.tokenizer)
    trace = serializer.serialize(
        nodes,
        metadata={
            **dict(trace_metadata or {}),
            "input_format": input_format,
            "trace_adapter": trace_adapter.name,
        },
    )
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
    attribution_method = _build_method(
        method,
        execution_model_name=execution_model_name,
        integrated_gradients_steps=integrated_gradients_steps,
    )
    attribution = attribution_method.attribute_objective(trace, resolved_objective, model)
    distributions = aggregate_attribution(trace, attribution, topk=topk_mean_k)
    rankings = tuple(
        AnalysisRanking(
            grain=distribution.grain,
            view_name=distribution.view_name,
            items=rank_distribution(distribution),
        )
        for distribution in distributions
    )
    _require_distribution(distributions, grain=ranking_grain, view_name=ranking_view)
    return SingleTraceAnalysisResult(
        trace=trace,
        target=target,
        attribution=attribution,
        distributions=distributions,
        rankings=rankings,
        metadata={
            "method": method,
            "ranking_grain": ranking_grain,
            "ranking_view": ranking_view,
            "topk_mean_k": topk_mean_k,
            "input_format": input_format,
            "trace_adapter": trace_adapter.name,
            "objective_type": resolved_objective.objective_type,
        },
        objective=resolved_objective,
    )


def write_analysis_json(result: SingleTraceAnalysisResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def analysis_to_dict(result: SingleTraceAnalysisResult) -> dict[str, Any]:
    return {
        "trace": {
            "tokenizer_name": result.trace.tokenizer_name,
            "token_count": max((span.end_token for span in result.trace.spans), default=0),
            "metadata": dict(result.trace.metadata),
        },
        "target": {
            "target_id": result.target.target_id,
            "node_ids": list(result.target.node_ids),
            "span": list(result.target.span) if result.target.span is not None else None,
        },
        "objective": target_objective_to_dict(result.objective) if result.objective is not None else None,
        "attribution": {
            "method_name": result.attribution.method_name,
            "attribution_model_name": result.attribution.attribution_model_name,
            "execution_model_name": result.attribution.execution_model_name,
            "same_model": result.attribution.same_model,
            "target_id": result.attribution.target_id,
            "token_scores": list(result.attribution.token_scores),
            "metadata": dict(result.attribution.metadata),
        },
        "distributions": [_distribution_to_dict(distribution) for distribution in result.distributions],
        "rankings": [_ranking_to_dict(ranking) for ranking in result.rankings],
        "metadata": dict(result.metadata),
    }


def _build_method(
    method: str,
    *,
    execution_model_name: str | None,
    integrated_gradients_steps: int,
) -> AttributionMethod:
    if method == "gradient_saliency":
        return GradientSaliencyAttribution(execution_model_name=execution_model_name)
    if method == "gradient_times_input":
        return GradientTimesInputAttribution(execution_model_name=execution_model_name)
    if method == "integrated_gradients":
        return IntegratedGradientsAttribution(
            execution_model_name=execution_model_name,
            steps=integrated_gradients_steps,
        )
    raise ValueError(
        "unknown attribution method "
        f"{method!r}; expected gradient_saliency, gradient_times_input, or integrated_gradients"
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
        raise ValueError("single-trace analysis currently expects exactly one failure target")
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


def _require_distribution(
    distributions: Sequence[AttributionDistribution],
    *,
    grain: str,
    view_name: str,
) -> AttributionDistribution:
    for distribution in distributions:
        if distribution.grain == grain and distribution.view_name == view_name:
            return distribution
    raise ValueError(f"no distribution for grain={grain!r}, view_name={view_name!r}")


def _distribution_to_dict(distribution: AttributionDistribution) -> dict[str, Any]:
    return {
        "view_name": distribution.view_name,
        "grain": distribution.grain,
        "target_id": distribution.target_id,
        "method_name": distribution.method_name,
        "attribution_model_name": distribution.attribution_model_name,
        "distribution_stats": dict(distribution.distribution_stats),
        "instances": [
            {
                "instance_id": instance.instance_id,
                "block_role": instance.block_role,
                "sub_block_kind": instance.sub_block_kind,
                "node_ids": list(instance.node_ids),
                "token_count": instance.token_count,
                "views": dict(instance.views),
            }
            for instance in distribution.instances
        ],
    }


def _ranking_to_dict(ranking: AnalysisRanking) -> dict[str, Any]:
    return {
        "grain": ranking.grain,
        "view_name": ranking.view_name,
        "items": [
            {
                "rank": item.rank,
                "score": item.score,
                "instance_id": item.instance.instance_id,
                "block_role": item.instance.block_role,
                "sub_block_kind": item.instance.sub_block_kind,
                "node_ids": list(item.instance.node_ids),
            }
            for item in ranking.items
        ],
    }
