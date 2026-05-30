"""Attribution method extension surface."""

from __future__ import annotations

from typing import Protocol

from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.objective import TargetObjective
from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.schema import SerializedTrace


class AttributionMethod(Protocol):
    name: str

    def attribute(
        self,
        trace: SerializedTrace,
        target: FailureTarget,
        model: ModelAdapter,
        *,
        contrastive_target: FailureTarget | None = None,
    ) -> AttributionResult: ...

    def attribute_objective(
        self,
        trace: SerializedTrace,
        objective: TargetObjective,
        model: ModelAdapter,
    ) -> AttributionResult: ...
