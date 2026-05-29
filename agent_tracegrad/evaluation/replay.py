"""Replay hook extension surface."""

from __future__ import annotations

from typing import Protocol

from agent_tracegrad.evaluation.spec import PerturbationSpec
from agent_tracegrad.trace.schema import SerializedTrace


class ReplayHook(Protocol):
    name: str

    def replay(
        self,
        perturbation_spec: PerturbationSpec,
        base_trace: SerializedTrace | None = None,
    ) -> SerializedTrace: ...
