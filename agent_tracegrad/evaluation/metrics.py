"""Evaluation metrics over attribution rankings and labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from statistics import median
from types import MappingProxyType
from typing import Mapping, Sequence

from agent_tracegrad.analysis.aggregation import AttributionDistribution
from agent_tracegrad.analysis.ranking import RankedAttribution, rank_distribution
from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel


@dataclass(frozen=True)
class MetricResult:
    metric_name: str
    value: float
    metadata: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def recall_at_k(
    ranking: Sequence[RankedAttribution],
    label: GroundTruthLabel,
    *,
    k: int,
) -> MetricResult:
    if k < 1:
        raise ValueError("k must be positive")
    target_ids = set(label.target_node_ids)
    if not target_ids:
        raise ValueError("label must contain target_node_ids")
    hits = 0
    for item in ranking[:k]:
        if target_ids.intersection(item.instance.node_ids):
            hits += 1
    value = hits / len(target_ids)
    return MetricResult(
        metric_name="recall@k",
        value=value,
        metadata={"k": k, "hits": hits, "target_count": len(target_ids), "label_id": label.label_id},
    )


def rank_correlation(
    ranking: Sequence[RankedAttribution],
    label: GroundTruthLabel,
) -> MetricResult:
    if len(ranking) < 2:
        return MetricResult(
            metric_name="rank_correlation",
            value=1.0,
            metadata={"label_id": label.label_id, "item_count": len(ranking)},
        )
    target_ids = set(label.target_node_ids)
    observed = [float(item.rank) for item in ranking]
    fallback_rank = float(len(ranking) + 1)
    expected = [
        1.0 if target_ids.intersection(item.instance.node_ids) else fallback_rank
        for item in ranking
    ]
    value = _spearman(expected, observed)
    return MetricResult(
        metric_name="rank_correlation",
        value=value,
        metadata={"label_id": label.label_id, "item_count": len(ranking)},
    )


def method_consistency(
    left: Sequence[RankedAttribution],
    right: Sequence[RankedAttribution],
    *,
    left_method_name: str,
    right_method_name: str,
) -> MetricResult:
    left_by_id = {item.instance.instance_id: float(item.rank) for item in left}
    right_by_id = {item.instance.instance_id: float(item.rank) for item in right}
    shared_ids = tuple(sorted(set(left_by_id).intersection(right_by_id)))
    if len(shared_ids) < 2:
        return MetricResult(
            metric_name="method_consistency",
            value=1.0,
            metadata={
                "left_method_name": left_method_name,
                "right_method_name": right_method_name,
                "shared_item_count": len(shared_ids),
            },
        )
    value = _spearman(
        [left_by_id[instance_id] for instance_id in shared_ids],
        [right_by_id[instance_id] for instance_id in shared_ids],
    )
    return MetricResult(
        metric_name="method_consistency",
        value=value,
        metadata={
            "left_method_name": left_method_name,
            "right_method_name": right_method_name,
            "shared_item_count": len(shared_ids),
        },
    )


def delta_ll_at_k(
    baseline_loss: float,
    perturbed_loss: float,
    ranking: Sequence[RankedAttribution],
    *,
    k: int,
    label_id: str,
    objective_id: str,
) -> MetricResult:
    if k < 1:
        raise ValueError("k must be positive")
    selected = ranking[:k]
    delta = float(perturbed_loss) - float(baseline_loss)
    return MetricResult(
        metric_name="delta_ll@k",
        value=delta,
        metadata={
            "k": k,
            "baseline_loss": float(baseline_loss),
            "perturbed_loss": float(perturbed_loss),
            "label_id": label_id,
            "objective_id": objective_id,
            "selected_count": len(selected),
            "selected_instance_ids": ",".join(item.instance.instance_id for item in selected),
        },
    )


def metrics_for_distribution(
    distribution: AttributionDistribution,
    label: GroundTruthLabel,
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> tuple[MetricResult, ...]:
    ranking = rank_distribution(distribution)
    results = [recall_at_k(ranking, label, k=k) for k in ks]
    results.append(rank_correlation(ranking, label))
    return tuple(results)


def summarize_metric_results(results: Sequence[MetricResult]) -> Mapping[str, float]:
    grouped: dict[str, list[float]] = {}
    for result in results:
        if isnan(result.value):
            continue
        grouped.setdefault(result.metric_name, []).append(result.value)
    return MappingProxyType(
        {
            metric_name: median(values)
            for metric_name, values in grouped.items()
            if values
        }
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("spearman inputs must have the same length")
    if len(left) < 2:
        return 1.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    denominator = (left_variance * right_variance) ** 0.5
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
