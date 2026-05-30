"""Framework-owned trace data structures and role/kind validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence

BLOCK_ROLES = frozenset({"system", "user", "agent"})

ROLE_TO_SUB_BLOCK_KINDS = MappingProxyType(
    {
        "system": frozenset({"system.instruction", "system.tool_schema", "system.skills"}),
        "user": frozenset({"user.content", "user.tool_result"}),
        "agent": frozenset({"agent.reasoning", "agent.content", "agent.tool_call"}),
    }
)

SUB_BLOCK_KINDS = frozenset(kind for kinds in ROLE_TO_SUB_BLOCK_KINDS.values() for kind in kinds)


def validate_block_role(block_role: str) -> str:
    if block_role not in BLOCK_ROLES:
        allowed = ", ".join(sorted(BLOCK_ROLES))
        raise ValueError(f"invalid block_role {block_role!r}; expected one of: {allowed}")
    return block_role


def validate_sub_block_kind(sub_block_kind: str) -> str:
    if sub_block_kind not in SUB_BLOCK_KINDS:
        allowed = ", ".join(sorted(SUB_BLOCK_KINDS))
        raise ValueError(f"invalid sub_block_kind {sub_block_kind!r}; expected one of: {allowed}")
    return sub_block_kind


def validate_role_kind_pair(block_role: str, sub_block_kind: str) -> tuple[str, str]:
    validate_block_role(block_role)
    validate_sub_block_kind(sub_block_kind)
    if sub_block_kind not in ROLE_TO_SUB_BLOCK_KINDS[block_role]:
        raise ValueError(f"sub_block_kind {sub_block_kind!r} is not valid for block_role {block_role!r}")
    return block_role, sub_block_kind


def _copy_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(metadata or {}))


@dataclass(frozen=True)
class TraceNode:
    node_id: str
    block_role: str
    sub_block_kind: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sequence_index: int | None = None
    timestamp: str | datetime | None = None
    parents: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        validate_role_kind_pair(self.block_role, self.sub_block_kind)
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.sequence_index is not None and self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative when provided")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))
        object.__setattr__(self, "parents", tuple(self.parents))


@dataclass(frozen=True)
class SpanMetadata:
    span_id: str
    node_id: str
    block_role: str
    sub_block_kind: str
    start_token: int
    end_token: int
    text_start_char: int | None = None
    text_end_char: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("span_id is required")
        if not self.node_id:
            raise ValueError("node_id is required")
        validate_role_kind_pair(self.block_role, self.sub_block_kind)
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token < self.start_token:
            raise ValueError("end_token must be greater than or equal to start_token")
        if self.text_start_char is not None and self.text_start_char < 0:
            raise ValueError("text_start_char must be non-negative when provided")
        if self.text_end_char is not None:
            if self.text_end_char < 0:
                raise ValueError("text_end_char must be non-negative when provided")
            if self.text_start_char is not None and self.text_end_char < self.text_start_char:
                raise ValueError("text_end_char must be greater than or equal to text_start_char")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True)
class SerializedTrace:
    nodes: Mapping[str, TraceNode]
    serialized_text: str
    spans: Sequence[SpanMetadata]
    tokenizer_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    token_offsets: Sequence[tuple[int, int]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name is required")
        if not isinstance(self.serialized_text, str):
            raise TypeError("serialized_text must be a string")
        nodes = dict(self.nodes)
        if not nodes:
            raise ValueError("nodes must not be empty")
        for node_id, node in nodes.items():
            if node_id != node.node_id:
                raise ValueError(f"nodes key {node_id!r} does not match TraceNode.node_id {node.node_id!r}")
        spans = tuple(self.spans)
        for span in spans:
            node = nodes.get(span.node_id)
            if node is None:
                raise ValueError(f"span {span.span_id!r} references unknown node {span.node_id!r}")
            if span.block_role != node.block_role or span.sub_block_kind != node.sub_block_kind:
                raise ValueError(f"span {span.span_id!r} role/kind does not match node {span.node_id!r}")
        token_offsets = tuple((int(start), int(end)) for start, end in self.token_offsets)
        for start, end in token_offsets:
            if start < 0 or end < start:
                raise ValueError("token_offsets must be non-negative half-open ranges")
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "token_offsets", token_offsets)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))
