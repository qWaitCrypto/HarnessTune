"""Minimal evaluation runner over generated trace-level samples."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.analysis import (
    AnalysisRanking,
    RankedAttribution,
    SingleTraceAnalysisResult,
    analyze_trace,
)
from agent_tracegrad.evaluation.metrics import (
    MetricResult,
    delta_ll_at_k,
    metrics_for_distribution,
    summarize_metric_results,
)
from agent_tracegrad.evaluation.orchestration import TraceEvaluationContext, generate_evaluation_context
from agent_tracegrad.evaluation.perturbation.trace_level import apply_trace_level_perturbation
from agent_tracegrad.evaluation.sample_generation import TraceLevelSample
from agent_tracegrad.evaluation.spec import PerturbationSpec
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.objective import TargetObjective
from agent_tracegrad.trace.serializer import TraceSerializer


@dataclass(frozen=True)
class EvaluationSampleResult:
    sample: TraceLevelSample
    analysis: SingleTraceAnalysisResult
    metrics: Sequence[MetricResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))


@dataclass(frozen=True)
class AblationCurvePoint:
    k: int
    target_node_ids: Sequence[str]
    baseline_loss: float
    ablated_loss: float
    delta_loss: float
    analysis: SingleTraceAnalysisResult
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        object.__setattr__(self, "target_node_ids", tuple(self.target_node_ids))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class EvaluationRunResult:
    context: TraceEvaluationContext
    baseline_analysis: SingleTraceAnalysisResult
    sample_results: Sequence[EvaluationSampleResult]
    ablation_curve: Sequence[AblationCurvePoint]
    summary: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_results", tuple(self.sample_results))
        object.__setattr__(self, "ablation_curve", tuple(self.ablation_curve))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary or {})))


def run_trace_level_evaluation(
    raw_trace: Any,
    *,
    model: ModelAdapter,
    tokenizer: Any | None = None,
    input_format: str = "json-fixture",
    target_node_ids: Sequence[str] | None = None,
    target_marker: str | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    objective: TargetObjective | None = None,
    operator_configs: Sequence[Mapping[str, Any]],
    max_samples: int | None = None,
    method: str = "gradient_saliency",
    execution_model_name: str | None = None,
    topk_mean_k: int = 5,
    ranking_grain: str = "node",
    ranking_view: str = "sum",
    integrated_gradients_steps: int = 16,
    trace_metadata: Mapping[str, Any] | None = None,
    metric_ks: Sequence[int] = (1, 3, 5),
    ablation_ks: Sequence[int] = (1, 3, 5),
    ablation_placeholder: str = "[ABLATE]",
) -> EvaluationRunResult:
    context = generate_evaluation_context(
        raw_trace,
        tokenizer=tokenizer or model.tokenizer,
        input_format=input_format,
        target_node_ids=target_node_ids,
        target_marker=target_marker,
        target_id=target_id,
        target_span=target_span,
        objective=objective,
        operator_configs=operator_configs,
        max_samples=max_samples,
        trace_metadata=trace_metadata,
    )
    sample_results: list[EvaluationSampleResult] = []
    baseline_analysis = analyze_trace(
        _trace_to_raw_payload(context.trace),
        input_format="json-fixture",
        target_node_ids=context.targets[0].node_ids,
        model=model,
        tokenizer=tokenizer or model.tokenizer,
        method=method,
        execution_model_name=execution_model_name,
        target_id=context.targets[0].target_id,
        target_span=context.targets[0].span,
        objective=context.objective,
        topk_mean_k=topk_mean_k,
        ranking_grain=ranking_grain,
        ranking_view=ranking_view,
        integrated_gradients_steps=integrated_gradients_steps,
        trace_metadata={
            **dict(trace_metadata or {}),
            "evaluation_sample_label_id": "baseline",
        },
    )
    baseline_distribution = _select_distribution(
        baseline_analysis.rankings,
        baseline_analysis.distributions,
        ranking_grain,
        ranking_view,
    )
    baseline_ranking = _select_ranking(baseline_analysis.rankings, ranking_grain, ranking_view)
    baseline_loss = _analysis_loss(baseline_analysis)
    ablation_curve = _run_ablation_curve(
        context,
        baseline_analysis=baseline_analysis,
        baseline_ranking=baseline_ranking,
        baseline_loss=baseline_loss,
        model=model,
        tokenizer=tokenizer or model.tokenizer,
        method=method,
        execution_model_name=execution_model_name,
        topk_mean_k=topk_mean_k,
        ranking_grain=ranking_grain,
        ranking_view=ranking_view,
        integrated_gradients_steps=integrated_gradients_steps,
        trace_metadata=trace_metadata,
        ablation_ks=ablation_ks,
        ablation_placeholder=ablation_placeholder,
    )
    for sample in context.samples:
        analysis = analyze_trace(
            _trace_to_raw_payload(sample.perturbation.trace),
            input_format="json-fixture",
            target_node_ids=context.targets[0].node_ids,
            model=model,
            tokenizer=tokenizer or model.tokenizer,
            method=method,
            execution_model_name=execution_model_name,
            target_id=context.targets[0].target_id,
            target_span=context.targets[0].span,
            objective=context.objective,
            topk_mean_k=topk_mean_k,
            ranking_grain=ranking_grain,
            ranking_view=ranking_view,
            integrated_gradients_steps=integrated_gradients_steps,
            trace_metadata={
                **dict(trace_metadata or {}),
                "evaluation_sample_label_id": sample.perturbation.label.label_id,
            },
        )
        distribution = _select_distribution(analysis.rankings, analysis.distributions, ranking_grain, ranking_view)
        metrics = list(metrics_for_distribution(distribution, sample.perturbation.label, ks=metric_ks))
        perturbed_loss = _analysis_loss(analysis)
        if baseline_loss is not None and perturbed_loss is not None:
            metrics.extend(
                delta_ll_at_k(
                    baseline_loss,
                    perturbed_loss,
                    baseline_ranking,
                    k=k,
                    label_id=sample.perturbation.label.label_id,
                    objective_id=context.objective.objective_id,
                )
                for k in metric_ks
            )
        sample_results.append(EvaluationSampleResult(sample=sample, analysis=analysis, metrics=metrics))

    return EvaluationRunResult(
        context=context,
        baseline_analysis=baseline_analysis,
        sample_results=sample_results,
        ablation_curve=ablation_curve,
        summary=summarize_metric_results(
            [metric for sample_result in sample_results for metric in sample_result.metrics]
            + [_ablation_metric(point) for point in ablation_curve]
        ),
    )


def _run_ablation_curve(
    context: TraceEvaluationContext,
    *,
    baseline_analysis: SingleTraceAnalysisResult,
    baseline_ranking: Sequence[RankedAttribution],
    baseline_loss: float | None,
    model: ModelAdapter,
    tokenizer: Any,
    method: str,
    execution_model_name: str | None,
    topk_mean_k: int,
    ranking_grain: str,
    ranking_view: str,
    integrated_gradients_steps: int,
    trace_metadata: Mapping[str, Any] | None,
    ablation_ks: Sequence[int],
    ablation_placeholder: str,
) -> tuple[AblationCurvePoint, ...]:
    del baseline_analysis
    if baseline_loss is None:
        return ()
    serializer = TraceSerializer(tokenizer)
    points: list[AblationCurvePoint] = []
    for k in _normalize_positive_ks(ablation_ks):
        target_node_ids = _top_ranked_node_ids(baseline_ranking, k=k)
        if not target_node_ids:
            continue
        perturbation = apply_trace_level_perturbation(
            context.trace,
            PerturbationSpec(
                operator="replace_with_placeholder",
                target_node_ids=target_node_ids,
                parameters={"placeholder": ablation_placeholder},
            ),
            serializer,
        )
        analysis = analyze_trace(
            _trace_to_raw_payload(perturbation.trace),
            input_format="json-fixture",
            target_node_ids=context.targets[0].node_ids,
            model=model,
            tokenizer=tokenizer,
            method=method,
            execution_model_name=execution_model_name,
            target_id=context.targets[0].target_id,
            target_span=context.targets[0].span,
            objective=context.objective,
            topk_mean_k=topk_mean_k,
            ranking_grain=ranking_grain,
            ranking_view=ranking_view,
            integrated_gradients_steps=integrated_gradients_steps,
            trace_metadata={
                **dict(trace_metadata or {}),
                "evaluation_sample_label_id": f"ablation@{k}",
            },
        )
        ablated_loss = _analysis_loss(analysis)
        if ablated_loss is None:
            continue
        points.append(
            AblationCurvePoint(
                k=k,
                target_node_ids=target_node_ids,
                baseline_loss=baseline_loss,
                ablated_loss=ablated_loss,
                delta_loss=ablated_loss - baseline_loss,
                analysis=analysis,
                metadata={
                    "operator": "replace_with_placeholder",
                    "placeholder": ablation_placeholder,
                    "ranking_grain": ranking_grain,
                    "ranking_view": ranking_view,
                },
            )
        )
    return tuple(points)


def _trace_to_raw_payload(trace) -> Mapping[str, Any]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "block_role": node.block_role,
                "sub_block_kind": node.sub_block_kind,
                "content": node.content,
                "metadata": dict(node.metadata),
                "sequence_index": node.sequence_index,
                "timestamp": node.timestamp,
                "parents": list(node.parents),
            }
            for node in sorted(trace.nodes.values(), key=lambda item: (item.sequence_index or 0, item.node_id))
        ]
    }


def _select_distribution(
    rankings: Sequence[AnalysisRanking],
    distributions,
    grain: str,
    view_name: str,
):
    del rankings
    for distribution in distributions:
        if distribution.grain == grain and distribution.view_name == view_name:
            return distribution
    raise ValueError(f"missing distribution for grain={grain!r}, view_name={view_name!r}")


def _select_ranking(
    rankings: Sequence[AnalysisRanking],
    grain: str,
    view_name: str,
):
    for ranking in rankings:
        if ranking.grain == grain and ranking.view_name == view_name:
            return ranking.items
    raise ValueError(f"missing ranking for grain={grain!r}, view_name={view_name!r}")


def _analysis_loss(analysis: SingleTraceAnalysisResult) -> float | None:
    loss = analysis.attribution.metadata.get("loss")
    if loss is None:
        return None
    return float(loss)


def _top_ranked_node_ids(ranking: Sequence[RankedAttribution], *, k: int) -> tuple[str, ...]:
    node_ids: list[str] = []
    for item in ranking[:k]:
        for node_id in item.instance.node_ids:
            if node_id not in node_ids:
                node_ids.append(node_id)
    return tuple(node_ids)


def _normalize_positive_ks(ks: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    return normalized


def _ablation_metric(point: AblationCurvePoint) -> MetricResult:
    return MetricResult(
        metric_name="ablation_delta_ll@k",
        value=point.delta_loss,
        metadata={
            "k": point.k,
            "baseline_loss": point.baseline_loss,
            "ablated_loss": point.ablated_loss,
            "target_node_ids": ",".join(point.target_node_ids),
        },
    )
