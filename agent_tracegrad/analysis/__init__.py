"""Analysis utilities for aggregating and ranking attribution evidence."""

from agent_tracegrad.analysis.aggregation import (
    AGGREGATION_GRAINS,
    AGGREGATION_VIEWS,
    AttributionDistribution,
    SubBlockAttribution,
    aggregate_attribution,
)
from agent_tracegrad.analysis.ranking import RankedAttribution, rank_distribution
from agent_tracegrad.analysis.single_trace import (
    AnalysisRanking,
    SingleTraceAnalysisResult,
    analysis_to_dict,
    analyze_normalized_trace,
    analyze_trace,
    write_analysis_json,
)

__all__ = [
    "AGGREGATION_GRAINS",
    "AGGREGATION_VIEWS",
    "AttributionDistribution",
    "AnalysisRanking",
    "RankedAttribution",
    "SingleTraceAnalysisResult",
    "SubBlockAttribution",
    "aggregate_attribution",
    "analysis_to_dict",
    "analyze_normalized_trace",
    "analyze_trace",
    "rank_distribution",
    "write_analysis_json",
]
