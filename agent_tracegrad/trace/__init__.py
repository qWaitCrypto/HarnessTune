"""Trace schemas, adapters, and serialization primitives."""

from agent_tracegrad.trace.adapter import TraceAdapter
from agent_tracegrad.trace.agentpi_adapter import AgentPIRawTraceAdapter
from agent_tracegrad.trace.json_adapter import JsonTraceAdapter
from agent_tracegrad.trace.registry import get_trace_adapter, trace_adapter_names
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
from agent_tracegrad.trace.serializer import OffsetTokenizer, TraceSerializer

__all__ = [
    "BLOCK_ROLES",
    "AgentPIRawTraceAdapter",
    "JsonTraceAdapter",
    "OffsetTokenizer",
    "SUB_BLOCK_KINDS",
    "ROLE_TO_SUB_BLOCK_KINDS",
    "SpanMetadata",
    "SerializedTrace",
    "TraceAdapter",
    "TraceSerializer",
    "TraceNode",
    "get_trace_adapter",
    "trace_adapter_names",
    "validate_block_role",
    "validate_role_kind_pair",
    "validate_sub_block_kind",
]
