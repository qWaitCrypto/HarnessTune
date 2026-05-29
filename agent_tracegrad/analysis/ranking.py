"""Rank aggregated attribution distributions for report and metric consumers."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.analysis.aggregation import AttributionDistribution, SubBlockAttribution


@dataclass(frozen=True)
class RankedAttribution:
    rank: int
    instance: SubBlockAttribution
    score: float


def rank_distribution(distribution: AttributionDistribution) -> tuple[RankedAttribution, ...]:
    ranked = sorted(
        distribution.instances,
        key=lambda instance: (-instance.views[distribution.view_name], instance.instance_id),
    )
    return tuple(
        RankedAttribution(rank=index + 1, instance=instance, score=instance.views[distribution.view_name])
        for index, instance in enumerate(ranked)
    )
