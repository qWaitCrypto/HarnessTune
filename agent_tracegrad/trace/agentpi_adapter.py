"""Adapter for AgentPI raw simulation traces."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from agent_tracegrad.trace.schema import TraceNode


class AgentPIRawTraceAdapter:
    """Convert AgentPI raw result records into canonical ``TraceNode`` objects."""

    name = "agentpi-raw"

    def adapt(self, raw_trace: Any) -> Sequence[TraceNode]:
        if not isinstance(raw_trace, Mapping):
            raise TypeError("AgentPI raw trace must be a mapping")
        simulation = raw_trace.get("simulation")
        if not isinstance(simulation, Mapping):
            raise ValueError("AgentPI raw trace must contain a 'simulation' mapping")

        nodes: list[TraceNode] = []
        sequence_index = 0
        policy = simulation.get("policy")
        if isinstance(policy, str) and policy:
            nodes.append(
                TraceNode(
                    node_id="agentpi:simulation:policy",
                    block_role="system",
                    sub_block_kind="system.instruction",
                    content=policy,
                    metadata=_trace_metadata(raw_trace, simulation, source_field="simulation.policy"),
                    sequence_index=sequence_index,
                    timestamp=simulation.get("start_time"),
                )
            )
            sequence_index += 1

        messages = simulation.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError("AgentPI raw trace simulation.messages must be a sequence")
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError("AgentPI raw message entries must be mappings")
            for node in _message_to_nodes(raw_trace, simulation, message, message_index, sequence_index):
                nodes.append(node)
                sequence_index += 1
        return tuple(nodes)


def _message_to_nodes(
    raw_trace: Mapping[str, Any],
    simulation: Mapping[str, Any],
    message: Mapping[str, Any],
    message_index: int,
    start_sequence_index: int,
) -> Sequence[TraceNode]:
    role = message.get("role")
    metadata = _message_metadata(raw_trace, simulation, message, message_index)
    timestamp = message.get("timestamp")
    nodes: list[TraceNode] = []
    locator = f"message-{message_index}"

    if role == "user":
        content = _string_content(message.get("content"))
        if content:
            nodes.append(
                TraceNode(
                    node_id=f"agentpi:{locator}:user-content",
                    block_role="user",
                    sub_block_kind="user.content",
                    content=content,
                    metadata=metadata,
                    sequence_index=start_sequence_index + len(nodes),
                    timestamp=timestamp,
                )
            )
    elif role == "tool":
        content = _string_content(message.get("content"))
        if content:
            nodes.append(
                TraceNode(
                    node_id=f"agentpi:{locator}:tool-result",
                    block_role="user",
                    sub_block_kind="user.tool_result",
                    content=content,
                    metadata=metadata,
                    sequence_index=start_sequence_index + len(nodes),
                    timestamp=timestamp,
                    parents=_parents_for_tool_result(message),
                )
            )
    elif role == "assistant":
        content = _string_content(message.get("content"))
        if content:
            nodes.append(
                TraceNode(
                    node_id=f"agentpi:{locator}:assistant-content",
                    block_role="agent",
                    sub_block_kind="agent.content",
                    content=content,
                    metadata=metadata,
                    sequence_index=start_sequence_index + len(nodes),
                    timestamp=timestamp,
                )
            )
        tool_calls = message.get("tool_calls")
        if tool_calls:
            nodes.append(
                TraceNode(
                    node_id=f"agentpi:{locator}:assistant-tool-call",
                    block_role="agent",
                    sub_block_kind="agent.tool_call",
                    content=_serialize_tool_calls(tool_calls),
                    metadata={**metadata, "tool_call_ids": _tool_call_ids(tool_calls)},
                    sequence_index=start_sequence_index + len(nodes),
                    timestamp=timestamp,
                )
            )
    return tuple(nodes)


def _trace_metadata(
    raw_trace: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    source_field: str,
) -> dict[str, Any]:
    task = raw_trace.get("task")
    task_id = task.get("id") if isinstance(task, Mapping) else raw_trace.get("task_id")
    reward_info = simulation.get("reward_info")
    return {
        "source_format": AgentPIRawTraceAdapter.name,
        "source_repo": raw_trace.get("source_repo"),
        "source_file": raw_trace.get("source_file"),
        "source_field": source_field,
        "simulation_id": simulation.get("id"),
        "simulation_index": raw_trace.get("simulation_index"),
        "task_id": task_id,
        "termination_reason": simulation.get("termination_reason"),
        "reward": reward_info.get("reward") if isinstance(reward_info, Mapping) else None,
        "review": simulation.get("review"),
        "user_only_review": simulation.get("user_only_review"),
        "hallucination_check": simulation.get("hallucination_check"),
    }


def _message_metadata(
    raw_trace: Mapping[str, Any],
    simulation: Mapping[str, Any],
    message: Mapping[str, Any],
    message_index: int,
) -> dict[str, Any]:
    metadata = _trace_metadata(raw_trace, simulation, source_field="simulation.messages")
    metadata.update(
        {
            "message_index": message_index,
            "turn_idx": message.get("turn_idx"),
            "role": message.get("role"),
            "message_id": message.get("id"),
            "requestor": message.get("requestor"),
            "error": message.get("error"),
            "cost": message.get("cost"),
            "usage": message.get("usage"),
        }
    )
    return metadata


def _string_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _serialize_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        return _string_content(tool_calls)
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, Mapping):
            normalized.append(
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "requestor": call.get("requestor"),
                }
            )
        else:
            normalized.append({"value": call})
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _tool_call_ids(tool_calls: Any) -> tuple[str, ...]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        return ()
    ids: list[str] = []
    for call in tool_calls:
        if isinstance(call, Mapping) and call.get("id") is not None:
            ids.append(str(call["id"]))
    return tuple(ids)


def _parents_for_tool_result(message: Mapping[str, Any]) -> tuple[str, ...]:
    call_id = message.get("id")
    if call_id is None:
        return ()
    return (str(call_id),)
