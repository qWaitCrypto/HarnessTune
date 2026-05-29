"""Generic failure target marker for the last explainable agent output."""

from __future__ import annotations

from typing import Sequence

from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.schema import SerializedTrace, TraceNode


class LastAgentOutputMarker:
    """Select the last ``agent.content`` node, falling back to ``agent.tool_call``."""

    name = "last-agent-output"

    def __init__(self, *, allow_tool_call_fallback: bool = True) -> None:
        self.allow_tool_call_fallback = allow_tool_call_fallback

    def mark(self, trace: SerializedTrace) -> Sequence[FailureTarget]:
        node = _last_agent_node(trace, "agent.content")
        if node is None and self.allow_tool_call_fallback:
            node = _last_agent_node(trace, "agent.tool_call")
        if node is None:
            return ()
        return (FailureTarget(target_id="last-agent-output", node_ids=(node.node_id,)),)


def _last_agent_node(trace: SerializedTrace, sub_block_kind: str) -> TraceNode | None:
    candidates = [
        node
        for node in trace.nodes.values()
        if node.block_role == "agent" and node.sub_block_kind == sub_block_kind
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda node: (node.sequence_index if node.sequence_index is not None else -1, node.node_id))
