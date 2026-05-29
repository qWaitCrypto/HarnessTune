"""Failure target data structure and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class FailureTarget:
    target_id: str
    node_ids: Sequence[str]
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id is required")
        node_ids = tuple(self.node_ids)
        if not node_ids:
            raise ValueError("node_ids must not be empty")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must not contain duplicates")
        object.__setattr__(self, "node_ids", node_ids)
        if self.span is not None:
            if len(self.span) != 2:
                raise ValueError("span must contain exactly two token indexes")
            start, end = int(self.span[0]), int(self.span[1])
            if start < 0:
                raise ValueError("span start must be non-negative")
            if end <= start:
                raise ValueError("span end must be greater than span start")
            object.__setattr__(self, "span", (start, end))

    def validate_against_trace(self, trace: SerializedTrace) -> None:
        target_token_positions: set[int] = set()
        for node_id in self.node_ids:
            node = trace.nodes.get(node_id)
            if node is None:
                raise ValueError(f"failure target references unknown node {node_id!r}")
            if node.block_role != "agent":
                raise ValueError(f"failure target node {node_id!r} must be in an agent block")
            for span in trace.spans:
                if span.node_id == node_id:
                    target_token_positions.update(range(span.start_token, span.end_token))
        if self.span is None:
            return
        start, end = self.span
        span_positions = set(range(start, end))
        if not span_positions.issubset(target_token_positions):
            raise ValueError("failure target span must fall inside the selected agent node token range")
