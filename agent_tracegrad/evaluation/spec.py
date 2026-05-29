"""Perturbation specification for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class PerturbationSpec:
    operator: str
    target_node_ids: Sequence[str]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operator:
            raise ValueError("operator is required")
        target_node_ids = tuple(self.target_node_ids)
        if not target_node_ids:
            raise ValueError("target_node_ids must not be empty")
        if len(set(target_node_ids)) != len(target_node_ids):
            raise ValueError("target_node_ids must not contain duplicates")
        object.__setattr__(self, "target_node_ids", target_node_ids)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters or {})))

    def validate_against_trace(self, trace: SerializedTrace) -> None:
        for node_id in self.target_node_ids:
            node = trace.nodes.get(node_id)
            if node is None:
                raise ValueError(f"perturbation references unknown node {node_id!r}")
            if node.block_role not in {"system", "user"}:
                raise ValueError(f"perturbation node {node_id!r} must be in a system or user block")
