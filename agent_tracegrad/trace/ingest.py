"""Thin orchestration helpers for adapting raw traces into canonical serialized traces."""

from __future__ import annotations

from typing import Any, Mapping

from agent_tracegrad.trace.adapter import TraceAdapter
from agent_tracegrad.trace.registry import get_trace_adapter
from agent_tracegrad.trace.schema import SerializedTrace
from agent_tracegrad.trace.serializer import TraceSerializer


def ingest_trace(
    raw_trace: Any,
    *,
    input_format: str = "json-fixture",
    tokenizer: Any,
    trace_metadata: Mapping[str, Any] | None = None,
    adapter: TraceAdapter | None = None,
) -> SerializedTrace:
    """Adapt a raw trace payload and serialize it into the canonical trace shape."""

    trace_adapter = adapter or get_trace_adapter(input_format)
    nodes = trace_adapter.adapt(raw_trace)
    serializer = TraceSerializer(tokenizer)
    return serializer.serialize(
        nodes,
        metadata={
            **dict(trace_metadata or {}),
            "input_format": input_format,
            "trace_adapter": trace_adapter.name,
        },
    )
