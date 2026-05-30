"""Rank aggregated attribution distributions for report and metric consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_tracegrad.analysis.aggregation import AttributionDistribution, SubBlockAttribution

RankBy = Literal["value", "abs", "positive", "negative"]


@dataclass(frozen=True)
class RankedAttribution:
    rank: int
    instance: SubBlockAttribution
    score: float


def rank_distribution(
    distribution: AttributionDistribution,
    *,
    rank_by: RankBy = "value",
) -> tuple[RankedAttribution, ...]:
    ranked = sorted(
        distribution.instances,
        key=lambda instance: (_ranking_key(instance.views[distribution.view_name], rank_by), instance.instance_id),
    )
    return tuple(
        RankedAttribution(rank=index + 1, instance=instance, score=instance.views[distribution.view_name])
        for index, instance in enumerate(ranked)
    )


def _ranking_key(score: float, rank_by: RankBy) -> float:
    if rank_by == "abs":
        return -abs(score)
    if rank_by == "positive":
        return -max(score, 0.0)
    if rank_by == "negative":
        return min(score, 0.0)
    return -score
