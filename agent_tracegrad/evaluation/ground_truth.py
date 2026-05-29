"""Ground-truth labels used by evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GroundTruthLabel:
    label_id: str
    target_node_ids: Sequence[str]
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label_id:
            raise ValueError("label_id is required")
        if not self.source:
            raise ValueError("source is required")
        target_node_ids = tuple(self.target_node_ids)
        if not target_node_ids:
            raise ValueError("target_node_ids must not be empty")
        object.__setattr__(self, "target_node_ids", target_node_ids)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
