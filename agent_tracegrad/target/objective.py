"""Diagnostic target objective data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.schema import SerializedTrace

ObjectiveType = Literal["bad_action", "expected_action", "contrastive"]
TargetSource = Literal["trace", "human", "benchmark", "synthetic"]


@dataclass(frozen=True)
class ExpectedTarget:
    target_id: str
    content: str
    source: TargetSource = "human"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("expected target_id is required")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("expected target content must be a non-empty string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class TargetObjective:
    objective_id: str
    objective_type: ObjectiveType
    bad_target: FailureTarget | None = None
    expected_target: ExpectedTarget | None = None
    source: TargetSource = "trace"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ValueError("objective_id is required")
        if self.objective_type not in {"bad_action", "expected_action", "contrastive"}:
            raise ValueError("unknown objective_type")
        if self.source not in {"trace", "human", "benchmark", "synthetic"}:
            raise ValueError("unknown target objective source")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
        _validate_objective_parts(self.objective_type, self.bad_target, self.expected_target, self.metadata)

    @classmethod
    def bad_action(
        cls,
        target: FailureTarget,
        *,
        objective_id: str | None = None,
        source: TargetSource = "trace",
        metadata: Mapping[str, Any] | None = None,
    ) -> "TargetObjective":
        return cls(
            objective_id=objective_id or target.target_id,
            objective_type="bad_action",
            bad_target=target,
            source=source,
            metadata=metadata or {},
        )

    @classmethod
    def expected_action(
        cls,
        expected: ExpectedTarget,
        *,
        objective_id: str | None = None,
        source: TargetSource | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TargetObjective":
        return cls(
            objective_id=objective_id or expected.target_id,
            objective_type="expected_action",
            expected_target=expected,
            source=source or expected.source,
            metadata=metadata or {},
        )

    @classmethod
    def contrastive(
        cls,
        bad_target: FailureTarget,
        expected_target: ExpectedTarget,
        *,
        objective_id: str | None = None,
        source: TargetSource = "human",
        metadata: Mapping[str, Any] | None = None,
    ) -> "TargetObjective":
        return cls(
            objective_id=objective_id or f"{bad_target.target_id}:vs:{expected_target.target_id}",
            objective_type="contrastive",
            bad_target=bad_target,
            expected_target=expected_target,
            source=source,
            metadata=metadata or {},
        )

    def validate_against_trace(self, trace: SerializedTrace) -> None:
        if self.bad_target is not None:
            self.bad_target.validate_against_trace(trace)


def target_objective_to_dict(objective: TargetObjective) -> dict[str, Any]:
    return {
        "objective_id": objective.objective_id,
        "objective_type": objective.objective_type,
        "source": objective.source,
        "bad_target": _failure_target_to_dict(objective.bad_target),
        "expected_target": _expected_target_to_dict(objective.expected_target),
        "metadata": dict(objective.metadata),
        "objective_formula": _objective_formula(objective),
    }


def _validate_objective_parts(
    objective_type: ObjectiveType,
    bad_target: FailureTarget | None,
    expected_target: ExpectedTarget | None,
    metadata: Mapping[str, Any],
) -> None:
    if objective_type == "bad_action" and bad_target is None:
        raise ValueError("bad_action objective requires bad_target")
    if objective_type == "expected_action" and expected_target is None:
        raise ValueError("expected_action objective requires expected_target")
    if objective_type == "contrastive" and expected_target is None:
        raise ValueError("contrastive objective requires expected_target")
    if (
        objective_type == "contrastive"
        and bad_target is None
        and metadata.get("requires_resolved_bad_target") is not True
    ):
        raise ValueError("contrastive objective requires bad_target and expected_target")


def _failure_target_to_dict(target: FailureTarget | None) -> dict[str, Any] | None:
    if target is None:
        return None
    return {
        "target_id": target.target_id,
        "node_ids": list(target.node_ids),
        "span": list(target.span) if target.span is not None else None,
    }


def _expected_target_to_dict(target: ExpectedTarget | None) -> dict[str, Any] | None:
    if target is None:
        return None
    return {
        "target_id": target.target_id,
        "content": target.content,
        "source": target.source,
        "metadata": dict(target.metadata),
    }


def _objective_formula(objective: TargetObjective) -> str:
    if objective.objective_type == "bad_action":
        return "log P(bad_target | context)"
    if objective.objective_type == "expected_action":
        return "log P(expected_target | context)"
    return "log P(bad_target | context) - log P(expected_target | context)"
