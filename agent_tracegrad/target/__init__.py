"""Failure target schema for agent-side attribution."""

from agent_tracegrad.target.last_agent_marker import LastAgentOutputMarker
from agent_tracegrad.target.marker import FailureTargetMarker
from agent_tracegrad.target.objective import (
    ExpectedTarget,
    TargetObjective,
    target_objective_to_dict,
)
from agent_tracegrad.target.registry import failure_target_marker_names, get_failure_target_marker
from agent_tracegrad.target.schema import FailureTarget

__all__ = [
    "ExpectedTarget",
    "FailureTarget",
    "FailureTargetMarker",
    "LastAgentOutputMarker",
    "TargetObjective",
    "failure_target_marker_names",
    "get_failure_target_marker",
    "target_objective_to_dict",
]
