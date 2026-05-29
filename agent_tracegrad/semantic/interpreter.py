"""Optional semantic interpretation Protocol and result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class SemanticInterpretation:
    interpreter_name: str
    interpreter_model_name: str | None
    target_id: str
    grain: str
    view_name: str
    summary: str
    instance_notes: Mapping[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.interpreter_name:
            raise ValueError("interpreter_name is required")
        if not self.target_id:
            raise ValueError("target_id is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "instance_notes", MappingProxyType(dict(self.instance_notes or {})))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


class SemanticInterpreter(Protocol):
    name: str

    def interpret(
        self,
        trace: SerializedTrace,
        target: FailureTarget,
        distributions: Sequence[Any],
    ) -> SemanticInterpretation: ...
