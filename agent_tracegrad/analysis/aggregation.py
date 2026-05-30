"""Aggregate token-level attribution scores to trace component distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from types import MappingProxyType
from typing import Mapping, Sequence

from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.trace.schema import SerializedTrace

AGGREGATION_GRAINS = ("node", "sub_block_kind")
AGGREGATION_VIEWS = (
    "sum",
    "mean",
    "length_norm",
    "topk_mean",
    "net_sum",
    "positive_sum",
    "negative_sum",
    "abs_sum",
    "topk_abs_mean",
)


@dataclass(frozen=True)
class SubBlockAttribution:
    instance_id: str
    block_role: str
    sub_block_kind: str
    node_ids: Sequence[str]
    token_count: int
    views: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.block_role not in {"system", "user"}:
            raise ValueError("SubBlockAttribution block_role must be system or user")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "views", MappingProxyType(dict(self.views)))


@dataclass(frozen=True)
class AttributionDistribution:
    view_name: str
    grain: str
    instances: Sequence[SubBlockAttribution]
    distribution_stats: Mapping[str, float]
    target_id: str
    method_name: str
    attribution_model_name: str

    def __post_init__(self) -> None:
        if self.view_name not in AGGREGATION_VIEWS:
            raise ValueError(f"unknown aggregation view {self.view_name!r}")
        if self.grain not in AGGREGATION_GRAINS:
            raise ValueError(f"unknown aggregation grain {self.grain!r}")
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "distribution_stats", MappingProxyType(dict(self.distribution_stats)))


@dataclass
class _Accumulator:
    instance_id: str
    block_role: str
    sub_block_kind: str
    node_ids: set[str] = field(default_factory=set)
    token_scores: list[float] = field(default_factory=list)


def aggregate_attribution(
    trace: SerializedTrace,
    result: AttributionResult,
    *,
    topk: int = 5,
) -> tuple[AttributionDistribution, ...]:
    """Build all node and sub-block-kind distributions for an attribution result."""

    if topk < 1:
        raise ValueError("topk must be positive")
    result.validate_against_trace(trace)
    node_instances = _build_instances(trace, result, grain="node", topk=topk)
    kind_instances = _build_instances(trace, result, grain="sub_block_kind", topk=topk)
    distributions: list[AttributionDistribution] = []
    for grain, instances in (("node", node_instances), ("sub_block_kind", kind_instances)):
        for view_name in AGGREGATION_VIEWS:
            distributions.append(
                AttributionDistribution(
                    view_name=view_name,
                    grain=grain,
                    instances=instances,
                    distribution_stats=_distribution_stats([item.views[view_name] for item in instances]),
                    target_id=result.target_id,
                    method_name=result.method_name,
                    attribution_model_name=result.attribution_model_name,
                )
            )
    return tuple(distributions)


def _build_instances(
    trace: SerializedTrace,
    result: AttributionResult,
    *,
    grain: str,
    topk: int,
) -> tuple[SubBlockAttribution, ...]:
    accumulators: dict[str, _Accumulator] = {}
    for span in trace.spans:
        if span.block_role == "agent":
            continue
        node = trace.nodes[span.node_id]
        instance_id = node.node_id if grain == "node" else node.sub_block_kind
        accumulator = accumulators.get(instance_id)
        if accumulator is None:
            accumulator = _Accumulator(
                instance_id=instance_id,
                block_role=node.block_role if grain == "node" else node.sub_block_kind.split(".", 1)[0],
                sub_block_kind=node.sub_block_kind,
            )
            accumulators[instance_id] = accumulator
        accumulator.node_ids.add(node.node_id)
        accumulator.token_scores.extend(result.token_scores[span.start_token : span.end_token])
    return tuple(_to_sub_block_attribution(item, topk=topk) for item in accumulators.values())


def _to_sub_block_attribution(accumulator: _Accumulator, *, topk: int) -> SubBlockAttribution:
    scores = tuple(accumulator.token_scores)
    return SubBlockAttribution(
        instance_id=accumulator.instance_id,
        block_role=accumulator.block_role,
        sub_block_kind=accumulator.sub_block_kind,
        node_ids=tuple(sorted(accumulator.node_ids)),
        token_count=len(scores),
        views=_views(scores, topk=topk),
    )


def _views(scores: Sequence[float], *, topk: int) -> Mapping[str, float]:
    if not scores:
        return {view: 0.0 for view in AGGREGATION_VIEWS}
    total = sum(scores)
    count = len(scores)
    k = min(topk, count)
    top_scores = sorted(scores, reverse=True)[:k]
    abs_scores = [abs(score) for score in scores]
    top_abs_scores = sorted(abs_scores, reverse=True)[:k]
    positive_sum = sum(score for score in scores if score > 0.0)
    negative_sum = sum(score for score in scores if score < 0.0)
    return {
        "sum": total,
        "mean": total / count,
        "length_norm": total / log(1.0 + count),
        "topk_mean": sum(top_scores) / k,
        "net_sum": total,
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "abs_sum": sum(abs_scores),
        "topk_abs_mean": sum(top_abs_scores) / k,
    }


def _distribution_stats(scores: Sequence[float]) -> Mapping[str, float]:
    if not scores:
        return {"entropy": 0.0, "top1_mass": 0.0, "top3_mass": 0.0, "gini": 0.0}
    if any(not isfinite(score) for score in scores):
        nan = float("nan")
        return {"entropy": nan, "top1_mass": nan, "top3_mass": nan, "gini": nan}
    abs_scores = [abs(score) for score in scores]
    total = sum(abs_scores)
    if total == 0.0:
        return {"entropy": 0.0, "top1_mass": 0.0, "top3_mass": 0.0, "gini": 0.0}
    probabilities = [score / total for score in abs_scores if score > 0.0]
    entropy = -sum(probability * log(probability) for probability in probabilities)
    sorted_probabilities = sorted((score / total for score in abs_scores), reverse=True)
    return {
        "entropy": entropy,
        "top1_mass": sorted_probabilities[0],
        "top3_mass": sum(sorted_probabilities[:3]),
        "gini": _gini(abs_scores),
        "positive_mass": sum(score for score in scores if score > 0.0) / total,
        "negative_mass": abs(sum(score for score in scores if score < 0.0)) / total,
        "net_direction": sum(scores) / total,
    }


def _gini(scores: Sequence[float]) -> float:
    total = sum(scores)
    if total == 0.0:
        return 0.0
    sorted_scores = sorted(scores)
    count = len(sorted_scores)
    weighted_sum = sum((index + 1) * score for index, score in enumerate(sorted_scores))
    return (2.0 * weighted_sum) / (count * total) - (count + 1.0) / count
