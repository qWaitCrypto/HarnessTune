"""Adapter for normalized JSON trace fixtures."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from agent_tracegrad.trace.schema import TraceNode


class JsonTraceAdapter:
    """Convert normalized JSON-like trace payloads into ``TraceNode`` objects."""

    name = "json-fixture"

    def adapt(self, raw_trace: Any) -> Sequence[TraceNode]:
        records = _extract_node_records(raw_trace)
        nodes = [_record_to_node(record, index) for index, record in enumerate(records)]
        return tuple(sorted(nodes, key=_node_sort_key))


def _extract_node_records(raw_trace: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(raw_trace, Mapping):
        raw_nodes = raw_trace.get("nodes")
        if raw_nodes is None:
            raise ValueError("normalized trace mapping must contain a 'nodes' field")
    else:
        raw_nodes = raw_trace
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        raise TypeError("normalized trace nodes must be a sequence of mappings")
    for record in raw_nodes:
        if not isinstance(record, Mapping):
            raise TypeError("each normalized trace node must be a mapping")
    return raw_nodes


def _record_to_node(record: Mapping[str, Any], fallback_index: int) -> TraceNode:
    try:
        node_id = record["node_id"]
        block_role = record["block_role"]
        sub_block_kind = record["sub_block_kind"]
        content = record["content"]
    except KeyError as exc:
        raise ValueError(f"normalized trace node missing required field {exc.args[0]!r}") from exc

    sequence_index = record.get("sequence_index", fallback_index)
    return TraceNode(
        node_id=str(node_id),
        block_role=str(block_role),
        sub_block_kind=str(sub_block_kind),
        content=content,
        metadata=record.get("metadata") or {},
        sequence_index=sequence_index,
        timestamp=record.get("timestamp"),
        parents=record.get("parents") or (),
    )


def _node_sort_key(node: TraceNode) -> tuple[int, str]:
    sequence_index = node.sequence_index if node.sequence_index is not None else 0
    return sequence_index, node.node_id
