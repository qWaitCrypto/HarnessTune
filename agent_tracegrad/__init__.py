"""Agent TraceGrad public package surface."""

from agent_tracegrad.trace.schema import (
    BLOCK_ROLES,
    SUB_BLOCK_KINDS,
    ROLE_TO_SUB_BLOCK_KINDS,
    SpanMetadata,
    SerializedTrace,
    TraceNode,
    validate_block_role,
    validate_role_kind_pair,
    validate_sub_block_kind,
)
from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.evaluation.spec import PerturbationSpec
from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel

__all__ = [
    "BLOCK_ROLES",
    "SUB_BLOCK_KINDS",
    "ROLE_TO_SUB_BLOCK_KINDS",
    "AttributionResult",
    "FailureTarget",
    "GroundTruthLabel",
    "PerturbationSpec",
    "SerializedTrace",
    "SpanMetadata",
    "TraceNode",
    "validate_block_role",
    "validate_role_kind_pair",
    "validate_sub_block_kind",
]
