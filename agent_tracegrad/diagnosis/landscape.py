"""Batch harness-level failure landscape analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.diagnosis.runner import read_diagnosis_json, run_diagnosis
from agent_tracegrad.diagnosis.types import DiagnosisResult
from agent_tracegrad.model.adapter import ModelAdapter

HARNESS_KINDS = frozenset({"system.instruction", "system.tool_schema", "system.skills"})


@dataclass(frozen=True)
class LandscapeTraceResult:
    trace_id: str
    trace_path: str | None
    fingerprint: Mapping[str, float]
    top_components: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", MappingProxyType(dict(self.fingerprint)))
        object.__setattr__(self, "top_components", tuple(self.top_components))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class LandscapeComponentStats:
    component_id: str
    sub_block_kind: str
    trace_count: int
    topk_count: int
    mean_score: float
    max_score: float
    min_score: float
    std_score: float


@dataclass(frozen=True)
class LandscapeCluster:
    cluster_id: str
    trace_ids: Sequence[str]
    top_components: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_ids", tuple(self.trace_ids))
        object.__setattr__(self, "top_components", tuple(self.top_components))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class LandscapeResult:
    traces: Sequence[LandscapeTraceResult]
    component_stats: Sequence[LandscapeComponentStats]
    clusters: Sequence[LandscapeCluster]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        object.__setattr__(self, "component_stats", tuple(self.component_stats))
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def run_landscape(
    trace_inputs: Sequence[tuple[str, Any, str | None]],
    *,
    model: ModelAdapter,
    expected_target_text: str | None = None,
    input_format: str = "json-fixture",
    target_node_ids: Sequence[str] | None = None,
    target_marker: str | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    method: str = "gradient_saliency",
    execution_model_name: str | None = None,
    topk_mean_k: int = 5,
    ranking_grain: str = "node",
    ranking_view: str = "sum",
    integrated_gradients_steps: int = 16,
    top_k: int = 3,
) -> LandscapeResult:
    if not trace_inputs:
        raise ValueError("landscape requires at least one trace")
    diagnoses: list[tuple[str, DiagnosisResult, str | None]] = []
    for trace_id, raw_trace, trace_path in trace_inputs:
        diagnoses.append(
            (
                trace_id,
                run_diagnosis(
                    raw_trace,
                    model=model,
                    expected_target_text=expected_target_text,
                    input_format=input_format,
                    target_node_ids=target_node_ids,
                    target_marker=target_marker,
                    target_id=target_id,
                    target_span=target_span,
                    method=method,
                    execution_model_name=execution_model_name,
                    topk_mean_k=topk_mean_k,
                    ranking_grain=ranking_grain,
                    ranking_view=ranking_view,
                    integrated_gradients_steps=integrated_gradients_steps,
                    trace_metadata={"trace_id": trace_id, "trace_path": trace_path},
                ),
                trace_path,
            )
        )
    return run_landscape_from_diagnoses(diagnoses, ranking_view=ranking_view, top_k=top_k)


def run_landscape_from_diagnoses(
    diagnoses: Sequence[tuple[str, DiagnosisResult, str | None]],
    *,
    ranking_view: str = "sum",
    top_k: int = 3,
) -> LandscapeResult:
    if not diagnoses:
        raise ValueError("landscape requires at least one diagnosis")
    trace_results: list[LandscapeTraceResult] = []
    for trace_id, diagnosis, trace_path in diagnoses:
        distribution = _select_harness_distribution(diagnosis, ranking_view=ranking_view)
        fingerprint = {
            contribution.instance_id: contribution.margin
            for contribution in distribution
            if contribution.sub_block_kind in HARNESS_KINDS
        }
        top_components = tuple(
            component
            for component, _ in sorted(fingerprint.items(), key=lambda item: (-abs(item[1]), item[0]))[:top_k]
        )
        trace_results.append(
            LandscapeTraceResult(
                trace_id=trace_id,
                trace_path=trace_path,
                fingerprint=fingerprint,
                top_components=top_components,
                metadata={"mode": diagnosis.metadata.get("mode", "unknown")},
            )
        )
    stats = _component_stats(trace_results, top_k=top_k)
    clusters = _cluster_by_top_component(trace_results)
    return LandscapeResult(
        traces=tuple(trace_results),
        component_stats=stats,
        clusters=clusters,
        metadata={
            "trace_count": len(trace_results),
            "top_k": top_k,
            "ranking_view": ranking_view,
            "source": "diagnosis-landscape",
        },
    )


def landscape_to_dict(result: LandscapeResult) -> dict[str, Any]:
    return {
        "metadata": dict(result.metadata),
        "traces": [
            {
                "trace_id": trace.trace_id,
                "trace_path": trace.trace_path,
                "fingerprint": dict(trace.fingerprint),
                "top_components": list(trace.top_components),
                "metadata": dict(trace.metadata),
            }
            for trace in result.traces
        ],
        "component_stats": [
            {
                "component_id": stat.component_id,
                "sub_block_kind": stat.sub_block_kind,
                "trace_count": stat.trace_count,
                "topk_count": stat.topk_count,
                "mean_score": stat.mean_score,
                "max_score": stat.max_score,
                "min_score": stat.min_score,
                "std_score": stat.std_score,
            }
            for stat in result.component_stats
        ],
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "trace_ids": list(cluster.trace_ids),
                "top_components": list(cluster.top_components),
                "metadata": dict(cluster.metadata),
            }
            for cluster in result.clusters
        ],
    }


def landscape_to_markdown(result: LandscapeResult) -> str:
    lines = [
        "# Agent TraceGrad Failure Landscape",
        "",
        "## Summary",
        "",
        f"- trace_count: {result.metadata['trace_count']}",
        f"- top_k: {result.metadata['top_k']}",
        "",
        "## Harness Components",
        "",
        "| Component | Kind | Traces | Top-k Count | Mean | Max | Std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stat in result.component_stats[:30]:
        lines.append(
            f"| `{stat.component_id}` | `{stat.sub_block_kind}` | {stat.trace_count} | "
            f"{stat.topk_count} | {stat.mean_score:.6g} | {stat.max_score:.6g} | {stat.std_score:.6g} |"
        )
    if result.clusters:
        lines.extend(["", "## Failure Mode Groups", ""])
        for cluster in result.clusters:
            top = ", ".join(f"`{item}`" for item in cluster.top_components)
            traces = ", ".join(f"`{item}`" for item in cluster.trace_ids)
            lines.append(f"- `{cluster.cluster_id}` traces={len(cluster.trace_ids)} top={top}: {traces}")
    return "\n".join(lines).rstrip() + "\n"


def write_landscape_json(result: LandscapeResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(landscape_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def write_landscape_markdown(result: LandscapeResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(landscape_to_markdown(result), encoding="utf-8")


def load_trace_inputs(path: str | Path) -> tuple[tuple[str, Any, str | None], ...]:
    root = Path(path)
    if root.is_file():
        return ((_trace_id(root), json.loads(root.read_text(encoding="utf-8")), str(root)),)
    files = sorted(item for item in root.iterdir() if item.is_file() and item.suffix == ".json")
    return tuple((_trace_id(item), json.loads(item.read_text(encoding="utf-8")), str(item)) for item in files)


def load_diagnosis_inputs(path: str | Path) -> tuple[tuple[str, DiagnosisResult, str | None], ...]:
    root = Path(path)
    if root.is_file():
        return ((_trace_id(root), read_diagnosis_json(root), str(root)),)
    files = sorted(
        item
        for item in root.iterdir()
        if item.is_file() and item.suffix == ".json" and not item.stem.endswith("-boundary")
    )
    return tuple((_trace_id(item), read_diagnosis_json(item), str(item)) for item in files)


def _select_harness_distribution(diagnosis, *, ranking_view: str):
    if diagnosis.margin_distributions:
        for distribution in diagnosis.margin_distributions:
            if distribution.grain == "node" and distribution.view_name == ranking_view:
                return distribution.contributions
    for distribution in diagnosis.bad_result.distributions:
        if distribution.grain == "node" and distribution.view_name == ranking_view:
            return tuple(
                _BadOnlyContribution(
                    instance_id=instance.instance_id,
                    sub_block_kind=instance.sub_block_kind,
                    margin=instance.views[ranking_view],
                )
                for instance in distribution.instances
            )
    raise ValueError(f"missing node/{ranking_view} distribution")


@dataclass(frozen=True)
class _BadOnlyContribution:
    instance_id: str
    sub_block_kind: str
    margin: float


def _component_stats(
    traces: Sequence[LandscapeTraceResult],
    *,
    top_k: int,
) -> tuple[LandscapeComponentStats, ...]:
    components = sorted({component for trace in traces for component in trace.fingerprint})
    stats: list[LandscapeComponentStats] = []
    for component in components:
        values = [trace.fingerprint.get(component, 0.0) for trace in traces]
        nonzero = [value for value in values if value != 0.0]
        topk_count = sum(1 for trace in traces if component in trace.top_components[:top_k])
        stats.append(
            LandscapeComponentStats(
                component_id=component,
                sub_block_kind=_component_kind(component, traces),
                trace_count=len(nonzero),
                topk_count=topk_count,
                mean_score=mean(values),
                max_score=max(values),
                min_score=min(values),
                std_score=pstdev(values) if len(values) > 1 else 0.0,
            )
        )
    return tuple(sorted(stats, key=lambda item: (-item.topk_count, -abs(item.mean_score), item.component_id)))


def _component_kind(component: str, traces: Sequence[LandscapeTraceResult]) -> str:
    if component.startswith("system."):
        return component
    for trace in traces:
        if component in trace.fingerprint:
            if "tool" in component:
                return "system.tool_schema"
            if "skill" in component:
                return "system.skills"
            return "system.instruction"
    return "system.instruction"


def _cluster_by_top_component(traces: Sequence[LandscapeTraceResult]) -> tuple[LandscapeCluster, ...]:
    groups: dict[str, list[LandscapeTraceResult]] = {}
    for trace in traces:
        key = trace.top_components[0] if trace.top_components else "no-harness-signal"
        groups.setdefault(key, []).append(trace)
    clusters: list[LandscapeCluster] = []
    for index, (component, members) in enumerate(sorted(groups.items()), start=1):
        component_counts: dict[str, int] = {}
        for member in members:
            for item in member.top_components:
                component_counts[item] = component_counts.get(item, 0) + 1
        top_components = tuple(
            item for item, _ in sorted(component_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
        )
        clusters.append(
            LandscapeCluster(
                cluster_id=f"cluster-{index}:{component}",
                trace_ids=tuple(member.trace_id for member in members),
                top_components=top_components,
                metadata={"grouping": "top_harness_component"},
            )
        )
    return tuple(clusters)


def _trace_id(path: Path) -> str:
    return path.stem
