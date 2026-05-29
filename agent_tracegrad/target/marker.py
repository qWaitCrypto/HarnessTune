"""Failure target marker extension surface."""

from __future__ import annotations

from typing import Protocol, Sequence

from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.schema import SerializedTrace


class FailureTargetMarker(Protocol):
    name: str

    def mark(self, trace: SerializedTrace) -> Sequence[FailureTarget]: ...
