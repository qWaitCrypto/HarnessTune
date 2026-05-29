"""Attribution result data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class AttributionResult:
    method_name: str
    attribution_model_name: str
    execution_model_name: str | None
    same_model: bool
    target_id: str
    token_scores: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method_name:
            raise ValueError("method_name is required")
        if not self.attribution_model_name:
            raise ValueError("attribution_model_name is required")
        if not self.target_id:
            raise ValueError("target_id is required")
        expected_same_model = (
            self.execution_model_name is not None and self.attribution_model_name == self.execution_model_name
        )
        if self.same_model != expected_same_model:
            raise ValueError("same_model must match attribution_model_name == execution_model_name")
        token_scores = tuple(float(score) for score in self.token_scores)
        if any(not isfinite(score) for score in token_scores):
            raise ValueError("token_scores must be finite")
        object.__setattr__(self, "token_scores", token_scores)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def validate_against_trace(self, trace: SerializedTrace) -> None:
        token_count = max((span.end_token for span in trace.spans), default=0)
        if len(self.token_scores) != token_count:
            raise ValueError("token_scores length must equal serialized token count")
        for span in trace.spans:
            if span.block_role == "agent":
                for position in range(span.start_token, span.end_token):
                    if self.token_scores[position] != 0.0:
                        raise ValueError("agent-range token scores must be zero")
