"""Tokenizer-aware trace serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from agent_tracegrad.trace.schema import SerializedTrace, SpanMetadata, TraceNode


class OffsetTokenizer(Protocol):
    name_or_path: str

    def __call__(self, text: str, *, return_offsets_mapping: bool, add_special_tokens: bool) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TraceSerializer:
    """Serialize trace nodes and build token span metadata from tokenizer offsets."""

    tokenizer: OffsetTokenizer
    tokenizer_name: str | None = None
    node_separator: str = "\n"

    def serialize(self, nodes: Sequence[TraceNode], *, metadata: Mapping[str, Any] | None = None) -> SerializedTrace:
        ordered_nodes = tuple(sorted(nodes, key=_node_sort_key))
        if not ordered_nodes:
            raise ValueError("nodes must not be empty")

        parts: list[str] = []
        char_ranges: dict[str, tuple[int, int]] = {}
        cursor = 0
        for index, node in enumerate(ordered_nodes):
            if index:
                parts.append(self.node_separator)
                cursor += len(self.node_separator)
            start = cursor
            parts.append(node.content)
            cursor += len(node.content)
            char_ranges[node.node_id] = (start, cursor)

        serialized_text = "".join(parts)
        encoded = self.tokenizer(serialized_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = tuple(encoded.get("offset_mapping") or ())
        if not offsets:
            raise ValueError("tokenizer must return non-empty offset_mapping")

        spans: list[SpanMetadata] = []
        for node in ordered_nodes:
            char_start, char_end = char_ranges[node.node_id]
            token_indexes = [
                token_index
                for token_index, offset in enumerate(offsets)
                if _token_overlaps_range(_coerce_offset(offset), char_start, char_end)
            ]
            if not token_indexes and node.content:
                raise ValueError(f"node {node.node_id!r} did not align to any tokenizer offsets")
            start_token = token_indexes[0] if token_indexes else _insertion_token_index(offsets, char_start)
            end_token = token_indexes[-1] + 1 if token_indexes else start_token
            spans.append(
                SpanMetadata(
                    span_id=f"span-{node.node_id}",
                    node_id=node.node_id,
                    block_role=node.block_role,
                    sub_block_kind=node.sub_block_kind,
                    start_token=start_token,
                    end_token=end_token,
                    text_start_char=char_start,
                    text_end_char=char_end,
                    metadata={"serializer": "trace-serializer"},
                )
            )

        return SerializedTrace(
            nodes={node.node_id: node for node in ordered_nodes},
            serialized_text=serialized_text,
            spans=tuple(spans),
            tokenizer_name=self.tokenizer_name or getattr(self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__),
            metadata=metadata or {},
        )


def _node_sort_key(node: TraceNode) -> tuple[int, str]:
    sequence_index = node.sequence_index if node.sequence_index is not None else 0
    return sequence_index, node.node_id


def _coerce_offset(offset: Any) -> tuple[int, int]:
    if not isinstance(offset, Sequence) or len(offset) != 2:
        raise ValueError("tokenizer offsets must be two-item sequences")
    start, end = int(offset[0]), int(offset[1])
    if start < 0 or end < start:
        raise ValueError("tokenizer offsets must be non-negative half-open ranges")
    return start, end


def _token_overlaps_range(offset: tuple[int, int], char_start: int, char_end: int) -> bool:
    token_start, token_end = offset
    if token_start == token_end:
        return False
    if char_start == char_end:
        return token_start == char_start
    return token_start < char_end and token_end > char_start


def _insertion_token_index(offsets: Sequence[Any], char_position: int) -> int:
    for token_index, offset in enumerate(offsets):
        start, _ = _coerce_offset(offset)
        if start >= char_position:
            return token_index
    return len(offsets)
