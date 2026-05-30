"""Export evaluation runner artifacts as JSON-ready records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent_tracegrad.analysis import analysis_to_dict
from agent_tracegrad.evaluation.evidence import build_evidence_report, evidence_report_to_dict
from agent_tracegrad.evaluation.metrics import MetricResult
from agent_tracegrad.evaluation.runner import AblationCurvePoint, EvaluationRunResult, EvaluationSampleResult
from agent_tracegrad.evaluation.sample_generation import TraceLevelSample
from agent_tracegrad.target.objective import target_objective_to_dict


def evaluation_run_to_dict(result: EvaluationRunResult) -> dict[str, Any]:
    """Convert an evaluation run result into a JSON-ready aggregate artifact."""

    return {
        "context": {
            "trace": _trace_summary(result.context.trace),
            "objective": target_objective_to_dict(result.context.objective),
            "targets": [
                {
                    "target_id": target.target_id,
                    "node_ids": list(target.node_ids),
                    "span": list(target.span) if target.span is not None else None,
                }
                for target in result.context.targets
            ],
            "metadata": dict(result.context.metadata),
        },
        "baseline_analysis": analysis_to_dict(result.baseline_analysis),
        "ablation_curve": [ablation_curve_point_to_dict(point) for point in result.ablation_curve],
        "summary": dict(result.summary),
        "sample_results": [
            evaluation_sample_result_to_dict(sample_result)
            for sample_result in result.sample_results
        ],
    }


def evaluation_sample_result_to_dict(result: EvaluationSampleResult) -> dict[str, Any]:
    evidence = build_evidence_report(result.analysis)
    return {
        "sample": trace_level_sample_to_dict(result.sample),
        "analysis": analysis_to_dict(result.analysis),
        "evidence": evidence_report_to_dict(evidence),
        "metrics": [metric_to_dict(metric) for metric in result.metrics],
    }


def evaluation_run_to_jsonl_records(result: EvaluationRunResult) -> tuple[dict[str, Any], ...]:
    """Return one JSON-ready record per evaluated sample."""

    return tuple(
        {
            "record_type": "evaluation_sample",
            "run_metadata": dict(result.context.metadata),
            "objective": target_objective_to_dict(result.context.objective),
            "summary": dict(result.summary),
            "ablation_curve": [ablation_curve_point_to_dict(point) for point in result.ablation_curve],
            **evaluation_sample_result_to_dict(sample_result),
        }
        for sample_result in result.sample_results
    )


def write_evaluation_artifacts(
    result: EvaluationRunResult,
    *,
    output_dir: str | Path,
    prefix: str = "tracegrad-evaluation",
) -> Mapping[str, Path]:
    """Write aggregate JSON and sample JSONL artifacts for an evaluation run."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    aggregate_path = path / f"{prefix}.json"
    jsonl_path = path / f"{prefix}.jsonl"
    markdown_path = path / f"{prefix}.md"
    aggregate_path.write_text(
        json.dumps(evaluation_run_to_dict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    jsonl_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in evaluation_run_to_jsonl_records(result)),
        encoding="utf-8",
    )
    markdown_path.write_text(evaluation_run_to_markdown(result), encoding="utf-8")
    return {"aggregate_json": aggregate_path, "samples_jsonl": jsonl_path, "markdown_report": markdown_path}


def evaluation_run_to_markdown(result: EvaluationRunResult) -> str:
    """Return a compact human-readable report for an evaluation run."""

    objective = target_objective_to_dict(result.context.objective)
    lines = [
        "# TraceGrad Evaluation Report",
        "",
        "## Objective",
        "",
        f"- objective_id: `{objective['objective_id']}`",
        f"- objective_type: `{objective['objective_type']}`",
        f"- formula: `{objective['objective_formula']}`",
        "",
        "## Summary",
        "",
    ]
    if result.summary:
        lines.extend(f"- {name}: {value:.6g}" for name, value in sorted(result.summary.items()))
    else:
        lines.append("- no metrics")
    if result.ablation_curve:
        lines.extend(["", "## Ablation Curve", ""])
        for point in result.ablation_curve:
            nodes = ", ".join(point.target_node_ids)
            lines.append(
                f"- k={point.k}: delta_loss={point.delta_loss:.6g}, "
                f"baseline={point.baseline_loss:.6g}, ablated={point.ablated_loss:.6g}, nodes=`{nodes}`"
            )
    lines.extend(["", "## Samples", ""])
    for index, sample_result in enumerate(result.sample_results, start=1):
        top_rank = _top_rank_item(sample_result)
        delta_metrics = [
            metric
            for metric in sample_result.metrics
            if metric.metric_name == "delta_ll@k"
        ]
        lines.append(f"### Sample {index}")
        lines.append("")
        lines.append(f"- operator: `{sample_result.sample.spec.operator}`")
        lines.append(f"- label_id: `{sample_result.sample.perturbation.label.label_id}`")
        if top_rank is not None:
            lines.append(f"- top_node: `{top_rank.instance.instance_id}` score={top_rank.score:.6g}")
        for metric in delta_metrics:
            lines.append(f"- delta_ll@{metric.metadata['k']}: {metric.value:.6g}")
        evidence = build_evidence_report(sample_result.analysis, top_tokens=5, top_windows=3)
        if evidence.top_windows:
            lines.append("")
            lines.append("Top windows:")
            for window in evidence.top_windows:
                lines.append(
                    f"- `{window.node_id}` score={window.score:.6g}: {window.text}"
                )
        if evidence.top_tokens:
            lines.append("")
            lines.append("Top tokens:")
            for token in evidence.top_tokens:
                text = token.text.replace("`", "\\`")
                lines.append(
                    f"- token {token.token_index} `{token.node_id}` score={token.score:.6g}: `{text}`"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def trace_level_sample_to_dict(sample: TraceLevelSample) -> dict[str, Any]:
    return {
        "spec": {
            "operator": sample.spec.operator,
            "target_node_ids": list(sample.spec.target_node_ids),
            "parameters": _json_ready_mapping(sample.spec.parameters),
        },
        "label": {
            "label_id": sample.perturbation.label.label_id,
            "target_node_ids": list(sample.perturbation.label.target_node_ids),
            "source": sample.perturbation.label.source,
            "metadata": _json_ready_mapping(sample.perturbation.label.metadata),
        },
        "perturbed_trace": _trace_summary(sample.perturbation.trace),
        "metadata": _json_ready_mapping(sample.metadata),
    }


def metric_to_dict(metric: MetricResult) -> dict[str, Any]:
    return {
        "metric_name": metric.metric_name,
        "value": metric.value,
        "metadata": _json_ready_mapping(metric.metadata),
    }


def ablation_curve_point_to_dict(point: AblationCurvePoint) -> dict[str, Any]:
    return {
        "k": point.k,
        "target_node_ids": list(point.target_node_ids),
        "baseline_loss": point.baseline_loss,
        "ablated_loss": point.ablated_loss,
        "delta_loss": point.delta_loss,
        "analysis": analysis_to_dict(point.analysis),
        "evidence": evidence_report_to_dict(build_evidence_report(point.analysis)),
        "metadata": _json_ready_mapping(point.metadata),
    }


def _top_rank_item(result: EvaluationSampleResult):
    for ranking in result.analysis.rankings:
        if ranking.grain == "node" and ranking.view_name == "sum" and ranking.items:
            return ranking.items[0]
    return None


def _trace_summary(trace) -> dict[str, Any]:
    return {
        "tokenizer_name": trace.tokenizer_name,
        "token_count": max((span.end_token for span in trace.spans), default=0),
        "node_count": len(trace.nodes),
        "metadata": _json_ready_mapping(trace.metadata),
    }


def _json_ready_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready_value(value) for key, value in dict(mapping).items()}


def _json_ready_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_ready_mapping(value)
    if isinstance(value, tuple | list):
        return [_json_ready_value(item) for item in value]
    return value
