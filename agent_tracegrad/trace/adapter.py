"""Trace adapter extension surface."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from agent_tracegrad.trace.schema import TraceNode


class TraceAdapter(Protocol):
    name: str

    def adapt(self, raw_trace: Any) -> Sequence[TraceNode]: ...
