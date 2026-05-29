"""Trace adapter registry."""

from __future__ import annotations

from agent_tracegrad.trace.adapter import TraceAdapter
from agent_tracegrad.trace.agentpi_adapter import AgentPIRawTraceAdapter
from agent_tracegrad.trace.json_adapter import JsonTraceAdapter


_ADAPTERS: dict[str, TraceAdapter] = {
    JsonTraceAdapter.name: JsonTraceAdapter(),
    "normalized": JsonTraceAdapter(),
    AgentPIRawTraceAdapter.name: AgentPIRawTraceAdapter(),
}


def get_trace_adapter(name: str) -> TraceAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unknown trace adapter {name!r}; expected one of: {allowed}") from exc


def trace_adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
