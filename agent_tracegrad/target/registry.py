"""Failure target marker registry."""

from __future__ import annotations

from agent_tracegrad.target.last_agent_marker import LastAgentOutputMarker
from agent_tracegrad.target.marker import FailureTargetMarker


_MARKERS: dict[str, FailureTargetMarker] = {
    LastAgentOutputMarker.name: LastAgentOutputMarker(),
}


def get_failure_target_marker(name: str) -> FailureTargetMarker:
    try:
        return _MARKERS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_MARKERS))
        raise ValueError(f"unknown failure target marker {name!r}; expected one of: {allowed}") from exc


def failure_target_marker_names() -> tuple[str, ...]:
    return tuple(sorted(_MARKERS))
